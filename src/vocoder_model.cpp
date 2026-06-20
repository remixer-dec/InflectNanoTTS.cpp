#include "vocoder_model.h"
#include "memory_trace.h"
#include <ggml-cpu.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>

namespace inflect {

static ggml_tensor* channel_param_3d(ggml_context* gctx, ggml_tensor* t) {
    return ggml_reshape_3d(gctx, t, 1, t->ne[0], 1);
}

static float tensor_get_f32(const ggml_tensor* t, int64_t i0, int64_t i1 = 0, int64_t i2 = 0) {
    const char* ptr = (const char*)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2];
    switch (t->type) {
        case GGML_TYPE_F32:
            return *(const float*)ptr;
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t*)ptr);
        default:
            break;
    }
    if (ggml_is_quantized(t->type)) {
        const auto* traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            fprintf(stderr, "[VocoderModel] unsupported quantized tensor read type %s for %s\n",
                    ggml_type_name(t->type), t->name);
            std::abort();
        }

        const char* row_ptr = (const char*)t->data + i1 * t->nb[1] + i2 * t->nb[2];
        struct QuantRowCache {
            const ggml_tensor* tensor = nullptr;
            const char* row = nullptr;
            std::vector<float> values;
        };
        thread_local QuantRowCache cache;
        if (cache.tensor != t || cache.row != row_ptr || (int64_t)cache.values.size() != t->ne[0]) {
            cache.tensor = t;
            cache.row = row_ptr;
            cache.values.resize(t->ne[0]);
            traits->to_float(row_ptr, cache.values.data(), t->ne[0]);
        }
        return cache.values[i0];
    }
    fprintf(stderr, "[VocoderModel] unsupported direct tensor read type %s for %s\n",
            ggml_type_name(t->type), t->name);
    std::abort();
}

static void tensor_set_f32(ggml_tensor* t, float v, int64_t i0, int64_t i1, int64_t i2) {
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "[VocoderModel] unsupported direct tensor write type %s for %s\n",
                ggml_type_name(t->type), t->name);
        std::abort();
    }
    *(float*)((char*)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2]) = v;
}

static void quant_conv1d_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata
) {
    const auto* p = static_cast<const VocoderQuantConv1dOpData*>(userdata);
    const ggml_tensor* x = dst->src[0];      // [T, in_ch, B]
    const ggml_tensor* weight = dst->src[1]; // [K*in_ch padded, out_ch]
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_quantized(weight->type)) {
        fprintf(stderr, "[VocoderModel] unsupported low-memory conv1d tensor types\n");
        std::abort();
    }

    const int64_t T = dst->ne[0];
    const int64_t out_ch = dst->ne[1];
    const int64_t batch = dst->ne[2];
    const int64_t in_ch = x->ne[1];

    const auto* traits = ggml_get_type_traits(weight->type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "[VocoderModel] unsupported quantized conv1d weight type %s\n",
                ggml_type_name(weight->type));
        std::abort();
    }

    thread_local std::vector<float> weight_row;
    weight_row.resize(weight->ne[0]);

    const char* x_data = static_cast<const char*>(x->data);
    char* dst_data = static_cast<char*>(dst->data);

    const int64_t o_start = (out_ch * ith) / nth;
    const int64_t o_end = (out_ch * (ith + 1)) / nth;
    for (int64_t o = o_start; o < o_end; o++) {
        const char* row_ptr = static_cast<const char*>(weight->data) + o * weight->nb[1];
        traits->to_float(row_ptr, weight_row.data(), weight->ne[0]);

        for (int64_t b = 0; b < batch; b++) {
            for (int64_t t = 0; t < T; t++) {
                float v = 0.0f;
                for (int64_t k = 0; k < p->kernel_size; k++) {
                    const int64_t src_t = t * p->stride + k * p->dilation - p->padding;
                    if (src_t < 0 || src_t >= x->ne[0]) {
                        continue;
                    }
                    for (int64_t i = 0; i < in_ch; i++) {
                        const int64_t flat = i * p->kernel_size + k;
                        const float xv = *reinterpret_cast<const float*>(
                            x_data + src_t * x->nb[0] + i * x->nb[1] + b * x->nb[2]);
                        v += xv * weight_row[flat];
                    }
                }
                *reinterpret_cast<float*>(dst_data + t * dst->nb[0] + o * dst->nb[1] + b * dst->nb[2]) = v;
            }
        }
    }
}

static ggml_tensor* add_channel_bias(ggml_context* gctx, ggml_tensor* x, ggml_tensor* bias) {
    ggml_tensor* b = channel_param_3d(gctx, bias);
    if (b->type != x->type && !ggml_is_quantized(b->type)) {
        b = ggml_cast(gctx, b, x->type);
    }
    return ggml_add(gctx, x, b);
}

static ggml_tensor* crop_time_3d(ggml_context* gctx, ggml_tensor* x, int left, int right) {
    if (left == 0 && right == 0) {
        return x;
    }
    const int64_t out_t = x->ne[0] - left - right;
    if (out_t <= 0) {
        fprintf(stderr, "[VocoderModel] invalid temporal crop\n");
        std::abort();
    }
    x = ggml_view_3d(gctx, x, out_t, x->ne[1], x->ne[2],
                     x->nb[1], x->nb[2], left * x->nb[0]);
    return ggml_cont(gctx, x);
}

static std::string debug_dump_dir() {
    const char* dir = std::getenv("INFLECT_DUMP_DIR");
    return (dir && dir[0]) ? std::string(dir) : std::string();
}

static void ensure_debug_dir(const std::string& dir) {
    if (dir.empty()) return;
    std::string cur;
    for (char c : dir) {
        cur.push_back(c);
        if (c == '/' && cur.size() > 1) {
            mkdir(cur.c_str(), 0755);
        }
    }
    mkdir(dir.c_str(), 0755);
}

static std::string debug_path(const std::string& dir, const std::string& name) {
    return dir + "/" + name;
}

static void mem_trace_top_graph_tensors(const char* label, ggml_cgraph* graph, int limit = 16) {
    if (!mem_trace_enabled()) return;
    const char* detail = std::getenv("INFLECT_MEM_TRACE_DETAIL");
    if (!detail || detail[0] != '1') return;

    struct Entry {
        size_t bytes;
        ggml_tensor* tensor;
    };
    std::vector<Entry> entries;
    const int n_nodes = ggml_graph_n_nodes(graph);
    entries.reserve(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        entries.push_back({ggml_nbytes(node), node});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.bytes > b.bytes;
    });

    const int n = std::min(limit, (int)entries.size());
    fprintf(stderr, "[mem] %s largest graph tensors:\n", label);
    for (int i = 0; i < n; i++) {
        const ggml_tensor* t = entries[i].tensor;
        const char* name = t->name[0] ? t->name : "(unnamed)";
        fprintf(stderr,
                "[mem]   %2d %8zu bytes %-12s %-24s shape=[%lld,%lld,%lld,%lld]\n",
                i + 1,
                entries[i].bytes,
                ggml_op_name(t->op),
                name,
                (long long)t->ne[0],
                (long long)t->ne[1],
                (long long)t->ne[2],
                (long long)t->ne[3]);
    }
}

static void debug_manifest(const std::string& dir, const std::string& line) {
    if (dir.empty()) return;
    std::ofstream f(debug_path(dir, "manifest.txt"), std::ios::app);
    f << line << "\n";
}

static void debug_save_f32(const std::string& dir, const std::string& name,
                           const std::vector<float>& data, const std::string& shape) {
    if (dir.empty()) return;
    std::ofstream f(debug_path(dir, name + ".f32"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    debug_manifest(dir, name + " f32 " + shape + " count=" + std::to_string(data.size()));
}

static std::vector<float> debug_transpose_time_channel(const std::vector<float>& data, int T, int C) {
    std::vector<float> out(C * T);
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < T; t++) {
            out[c * T + t] = data[t + c * T];
        }
    }
    return out;
}

static ggml_tensor* conv1d_vocoder(
    ggml_context* gctx,
    ggml_tensor* weight,
    ggml_tensor* x,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    std::vector<VocoderQuantConv1dOpData>& op_data
) {
    if (!ggml_is_quantized(weight->type)) {
        ggml_tensor* kernel = weight;
        if (kernel->type != GGML_TYPE_F16) {
            kernel = ggml_cast(gctx, kernel, GGML_TYPE_F16);
        }
        return ggml_conv_1d(gctx, kernel, x, stride, padding, dilation);
    }

    const int64_t in_ch = x->ne[1];
    const int64_t out_ch = weight->ne[1];
    const int64_t flat = kernel_size * in_ch;
    if (weight->ne[0] < flat) {
        fprintf(stderr, "[VocoderModel] quantized conv weight too small: weight=[%lld,%lld] flat=%lld\n",
                (long long)weight->ne[0], (long long)weight->ne[1], (long long)flat);
        std::abort();
    }

#if defined(INFLECT_LOW_MEMORY)
    const int64_t out_t = (x->ne[0] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    op_data.push_back({kernel_size, stride, padding, dilation});
    ggml_tensor* args[] = {x, weight};
    return ggml_custom_4d(gctx, GGML_TYPE_F32, out_t, out_ch, x->ne[2], 1,
                          args, 2, quant_conv1d_op, GGML_N_TASKS_MAX, &op_data.back());
#else
    // ggml_im2col only needs this tensor for shape/type; weights are consumed
    // by the padded quantized matrix multiply below. Use F32 columns because
    // this GGML CPU PAD op only supports F32.
    ggml_tensor* shape_kernel = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, kernel_size, in_ch, out_ch);
    ggml_tensor* im2col = ggml_im2col(gctx, shape_kernel, x, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor* cols = ggml_reshape_2d(gctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    if (weight->ne[0] > cols->ne[0]) {
        cols = ggml_pad(gctx, cols, weight->ne[0] - cols->ne[0], 0, 0, 0);
    }
    ggml_tensor* y = ggml_mul_mat(gctx, weight, cols);
    y = ggml_cont(gctx, ggml_transpose(gctx, y));
    return ggml_reshape_3d(gctx, y, im2col->ne[1], out_ch, im2col->ne[2]);
#endif
}

static void quant_conv_transpose1d_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata
) {
    const auto* p = static_cast<const QuantConvTranspose1dOpData*>(userdata);
    const ggml_tensor* x = dst->src[0];      // [T_in, in_ch, B]
    const ggml_tensor* weight = dst->src[1]; // [K*in_ch padded, out_ch]
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_quantized(weight->type)) {
        fprintf(stderr, "[VocoderModel] unsupported low-memory conv_transpose1d tensor types\n");
        std::abort();
    }

    const int64_t out_t = dst->ne[0];
    const int64_t out_ch = dst->ne[1];
    const int64_t batch = dst->ne[2];
    const int64_t in_t = x->ne[0];
    const int64_t in_ch = x->ne[1];

    const auto* traits = ggml_get_type_traits(weight->type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "[VocoderModel] unsupported quantized conv_transpose1d weight type %s\n",
                ggml_type_name(weight->type));
        std::abort();
    }

    thread_local std::vector<float> weight_row;
    weight_row.resize(weight->ne[0]);

    const char* x_data = static_cast<const char*>(x->data);
    char* dst_data = static_cast<char*>(dst->data);

    const int64_t o_start = (out_ch * ith) / nth;
    const int64_t o_end = (out_ch * (ith + 1)) / nth;
    for (int64_t o = o_start; o < o_end; o++) {
        const char* row_ptr = static_cast<const char*>(weight->data) + o * weight->nb[1];
        traits->to_float(row_ptr, weight_row.data(), weight->ne[0]);

        for (int64_t b = 0; b < batch; b++) {
            for (int64_t t = 0; t < out_t; t++) {
                float v = 0.0f;
                for (int64_t k = 0; k < p->kernel_size; k++) {
                    const int64_t rem = t - k;
                    if (rem < 0 || rem % p->stride != 0) {
                        continue;
                    }
                    const int64_t src_t = rem / p->stride;
                    if (src_t < 0 || src_t >= in_t) {
                        continue;
                    }
                    for (int64_t i = 0; i < in_ch; i++) {
                        const int64_t flat = k * in_ch + i;
                        const float xv = *reinterpret_cast<const float*>(
                            x_data + src_t * x->nb[0] + i * x->nb[1] + b * x->nb[2]);
                        v += xv * weight_row[flat];
                    }
                }
                *reinterpret_cast<float*>(dst_data + t * dst->nb[0] + o * dst->nb[1] + b * dst->nb[2]) = v;
            }
        }
    }
}

static ggml_tensor* quant_or_f16_conv_transpose_1d(
    ggml_context* ctx,
    ggml_tensor* weight,
    ggml_tensor* x,
    int kernel_size,
    int stride,
    std::vector<QuantConvTranspose1dOpData>& op_data
) {
    if (!ggml_is_quantized(weight->type)) {
        ggml_tensor* kernel = weight;
        if (kernel->type != GGML_TYPE_F16) {
            kernel = ggml_cast(ctx, kernel, GGML_TYPE_F16);
        }
        return ggml_conv_transpose_1d(ctx, kernel, x, stride, 0, 1);
    }

    const int64_t in_ch = x->ne[1];
    const int64_t out_ch = weight->ne[1];
    const int64_t flat = kernel_size * in_ch;
    if (weight->ne[0] < flat) {
        fprintf(stderr, "[VocoderModel] quantized upsample weight too small: weight=[%lld,%lld] flat=%lld\n",
                (long long)weight->ne[0], (long long)weight->ne[1], (long long)flat);
        std::abort();
    }

    const int64_t out_t = (x->ne[0] - 1) * stride + kernel_size;
    op_data.push_back({kernel_size, stride});
    ggml_tensor* args[] = {x, weight};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, out_t, out_ch, x->ne[2], 1,
                          args, 2, quant_conv_transpose1d_op, GGML_N_TASKS_MAX, &op_data.back());
}

// ═════════════════════════════════════════════════════════════════════════
// Construction
// ═════════════════════════════════════════════════════════════════════════

VocoderModel::VocoderModel(const VocoderConfig& config) : config_(config) {
    const int n_ups = config.upsample_rates.size();
    const int n_res_per_up = config.resblock_kernel_sizes.size();
    weights_.resblocks.resize(n_ups * n_res_per_up);
    weights_.ups_w.resize(n_ups);
    weights_.ups_b.resize(n_ups);
    weights_.up_acts_alpha.resize(n_ups);
}

VocoderModel::~VocoderModel() {
    // ModelLoader owns the weight context; this class only keeps tensor pointers.
}

int VocoderModel::total_upsample() const {
    int total = 1;
    for (int r : config_.upsample_rates) total *= r;
    return total;
}

// ═════════════════════════════════════════════════════════════════════════
// Weight norm folding
// ═════════════════════════════════════════════════════════════════════════

void VocoderModel::fold_weight_norm(
    ggml_tensor* dst,
    ggml_tensor* weight_v,
    ggml_tensor* weight_g,
    int dim0, int dim1, int dim2,
    bool is_transpose
) {
    // For Conv1d:
    //   weight_v: [K, in, out], weight_g: [1, 1, out]
    //   folded[o] = v[:, :, o] * (g[0,0,o] / ||v[:, :, o]||)
    //
    // For ConvTranspose1d:
    //   weight_v: [K, in, out], weight_g: [1, in, 1]
    //   folded[i] = v[:, i, :] * (g[0,i,0] / ||v[:, i, :]||)

    float* v_data = (float*)weight_v->data;
    float* g_data = (float*)weight_g->data;
    float* d_data = (float*)dst->data;

    if (!is_transpose) {
        // Conv1d: normalize per output channel
        int out_ch = dim2;
        int elements_per_out = dim0 * dim1; // K * in
        for (int o = 0; o < out_ch; o++) {
            float norm = 0.0f;
            for (int e = 0; e < elements_per_out; e++) {
                float val = v_data[o * elements_per_out + e];
                norm += val * val;
            }
            norm = std::sqrt(norm) + 1e-8f;
            float scale = g_data[o] / norm;
            for (int e = 0; e < elements_per_out; e++) {
                d_data[o * elements_per_out + e] =
                    v_data[o * elements_per_out + e] * scale;
            }
        }
    } else {
        // ConvTranspose1d: normalize per input channel
        int in_ch = dim1;
        int out_ch = dim2;
        for (int i = 0; i < in_ch; i++) {
            float norm = 0.0f;
            for (int k = 0; k < dim0; k++) {
                for (int o = 0; o < out_ch; o++) {
                    float val = v_data[k * in_ch * out_ch + i * out_ch + o];
                    norm += val * val;
                }
            }
            norm = std::sqrt(norm) + 1e-8f;
            float scale = g_data[i] / norm;
            for (int k = 0; k < dim0; k++) {
                for (int o = 0; o < out_ch; o++) {
                    int idx = k * in_ch * out_ch + i * out_ch + o;
                    d_data[idx] = v_data[idx] * scale;
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Weight loading
// ═════════════════════════════════════════════════════════════════════════

bool VocoderModel::load(ModelLoader& loader) {
    const int n_ups = config_.upsample_rates.size();
    const int n_res = config_.resblock_kernel_sizes.size();
    const int init_ch = config_.upsample_initial_channel;
    bool ok = true;

    auto get = [&](const std::string& name) -> ggml_tensor* {
        if (loader.has_tensor(name)) {
            return loader.get_tensor(name);
        }
        static const std::string prefix = "generator.";
        if (name.rfind(prefix, 0) == 0) {
            std::string stripped = name.substr(prefix.size());
            if (loader.has_tensor(stripped)) {
                return loader.get_tensor(stripped);
            }
        }
        fprintf(stderr, "[VocoderModel] Required tensor not found: %s\n", name.c_str());
        ok = false;
        return nullptr;
    };

    // ── Conv pre ────────────────────────────────────────────────────
    // Converter folds weight_norm, so runtime expects plain *.weight tensors.
    {
        weights_.conv_pre_w = get("generator.conv_pre.weight");
        weights_.conv_pre_b = get("generator.conv_pre.bias");
    }

    // ── Upsampling layers ───────────────────────────────────────────
    for (int i = 0; i < n_ups; i++) {
        std::string prefix = "generator.ups." + std::to_string(i);
        weights_.ups_w[i] = get(prefix + ".weight");
        weights_.ups_b[i] = get(prefix + ".bias");

        // Snake alpha
        std::string act_prefix = "generator.up_acts." + std::to_string(i);
        weights_.up_acts_alpha[i] = get(act_prefix + ".log_alpha");
    }

    // ── Residual blocks ─────────────────────────────────────────────
    for (int i = 0; i < n_ups; i++) {
        for (int j = 0; j < n_res; j++) {
            int rb_idx = i * n_res + j;
            auto& rb = weights_.resblocks[rb_idx];

            for (int c = 0; c < 3; c++) {
                // convs1: dilated conv, weight [K, out, out]
                std::string p1 = "generator.resblocks." + std::to_string(rb_idx) +
                    ".convs1." + std::to_string(c);
                rb.convs1_w.push_back(get(p1 + ".weight"));
                rb.convs1_b.push_back(get(p1 + ".bias"));

                // convs2: dilation=1 conv, weight [K, out, out]
                std::string p2 = "generator.resblocks." + std::to_string(rb_idx) +
                    ".convs2." + std::to_string(c);
                rb.convs2_w.push_back(get(p2 + ".weight"));
                rb.convs2_b.push_back(get(p2 + ".bias"));

                // Snake alphas
                rb.acts1_alpha.push_back(get(
                    "generator.resblocks." + std::to_string(rb_idx) +
                    ".acts1." + std::to_string(c) + ".log_alpha"));
                rb.acts2_alpha.push_back(get(
                    "generator.resblocks." + std::to_string(rb_idx) +
                    ".acts2." + std::to_string(c) + ".log_alpha"));
            }
        }
    }

    // ── Post layers ─────────────────────────────────────────────────
    {
        weights_.conv_post_w = get("generator.conv_post.weight");
        weights_.conv_post_b = get("generator.conv_post.bias");
        weights_.post_act_alpha = get("generator.post_act.log_alpha");
    }

    if (!ok) {
        fprintf(stderr, "[VocoderModel] Incomplete vocoder GGUF; regenerate with folded weight tensors.\n");
        return false;
    }

    wctx_ = loader.ctx();
    return true;
}

// ═════════════════════════════════════════════════════════════════════════
// Snake activation: x + sin²(α·x) / α
// ═════════════════════════════════════════════════════════════════════════

ggml_tensor* VocoderModel::snake(
    ggml_context* gctx,
    ggml_tensor* x,
    ggml_tensor* log_alpha
) {
    // α = exp(log_alpha), clamped to [1e-4, 100]
    ggml_tensor* alpha = ggml_exp(gctx, log_alpha);
    // Clamp: use min/max ops
    alpha = ggml_clamp(gctx, alpha, 1e-4f, 100.0f);
    alpha = channel_param_3d(gctx, alpha);

    // α·x
    ggml_tensor* ax = ggml_mul(gctx, x, alpha);

    // sin²(α·x) = (1 - cos(2α·x)) / 2
    // Using sin² directly: sin(ax)²
    ggml_tensor* sin_ax = ggml_sin(gctx, ax);
    ggml_tensor* sin_sq = ggml_sqr(gctx, sin_ax);

    // sin²(α·x) / α
    ggml_tensor* term = ggml_div(gctx, sin_sq, alpha);

    // x + sin²(α·x) / α
    return ggml_add(gctx, x, term);
}

// ═════════════════════════════════════════════════════════════════════════
// ResBlock1
// ═════════════════════════════════════════════════════════════════════════

ggml_tensor* VocoderModel::build_resblock(
    ggml_context* gctx,
    ggml_tensor* x,   // [T, ch, 1] — GGML conv1d input layout
    const ResBlockWeights& w,
    int kernel_size
) {
    // ResBlock1:
    //   for each (c1, c2, a1, a2) in zip(convs1, convs2, acts1, acts2):
    //     y = a1(x) → c1(y) → a2(y) → c2(y) → x = x + y

    for (int i = 0; i < 3; i++) {
        int K = kernel_size;
        int dilation = config_.resblock_dilation_sizes[i % config_.resblock_dilation_sizes.size()][i];
        int pad1 = (K * dilation - dilation) / 2;
        int pad2 = (K - 1) / 2; // dilation=1 for convs2

        // a1(x)
        ggml_tensor* y = snake(gctx, x, w.acts1_alpha[i]);
        // c1(y) — dilated conv
        y = conv1d_vocoder(gctx, w.convs1_w[i], y, K, 1, pad1, dilation, quant_conv1d_ops_);
        y = add_channel_bias(gctx, y, w.convs1_b[i]);

        // a2(y)
        y = snake(gctx, y, w.acts2_alpha[i]);
        // c2(y) — dilation=1 conv
        y = conv1d_vocoder(gctx, w.convs2_w[i], y, K, 1, pad2, 1, quant_conv1d_ops_);
        y = add_channel_bias(gctx, y, w.convs2_b[i]);

        // Residual
        x = ggml_add(gctx, x, y);
    }
    return x;
}

// ═════════════════════════════════════════════════════════════════════════
// Vocoder graph
// ═════════════════════════════════════════════════════════════════════════

ggml_cgraph* VocoderModel::build_vocoder_graph(
    ggml_context* gctx,
    ggml_tensor* mel  // [n_mels, n_frames, 1]
) {
    const int n_mels = config_.num_mels;
    const int n_ups = config_.upsample_rates.size();
    const int n_res = config_.resblock_kernel_sizes.size();
    const int init_ch = config_.upsample_initial_channel;
    (void)init_ch;

    const bool capture_debug = !debug_dump_dir().empty();
    std::vector<ggml_tensor*> debug_outputs;
    auto capture = [&](const std::string& name, ggml_tensor* t) {
        if (!capture_debug) return;
        ggml_tensor* out = ggml_cpy(gctx, t, ggml_dup_tensor(gctx, t));
        ggml_set_name(out, name.c_str());
        ggml_set_output(out);
        debug_outputs.push_back(out);
    };

    // ── Conv pre ────────────────────────────────────────────────────
    // GGML conv_1d expects input [T, in_ch, B] and weight [K, in_ch, out_ch]
    // Our mel is [n_mels, n_frames, 1] → need to permute to [n_frames, n_mels, 1]
    ggml_tensor* x = ggml_permute(gctx, mel, 1, 0, 2, 3); // [n_frames, n_mels, 1]
    x = ggml_cont(gctx, x);

    x = conv1d_vocoder(gctx, weights_.conv_pre_w, x, 7, 1, 3, 1, quant_conv1d_ops_);
    x = add_channel_bias(gctx, x, weights_.conv_pre_b);
    capture("vocoder_conv_pre", x);

    // ── Upsampling + ResBlocks ──────────────────────────────────────
    for (int i = 0; i < n_ups; i++) {
        int rate = config_.upsample_rates[i];
        int K = config_.upsample_kernel_sizes[i];
        int pad = (K - rate) / 2;

        // Snake activation before upsampling
        x = snake(gctx, x, weights_.up_acts_alpha[i]);

        // Transposed conv: upsampling
        // ggml_conv_transpose_1d(ctx, kernel, input, stride, padding, dilation)
        // This GGML revision only supports p0=0 and d0=1 for conv_transpose_1d.
        x = quant_or_f16_conv_transpose_1d(gctx, weights_.ups_w[i], x, K, rate, quant_conv_transpose_ops_);
        x = crop_time_3d(gctx, x, pad, pad);
        capture("vocoder_upsample_" + std::to_string(i), x);

        // Residual blocks (sum and average)
        ggml_tensor* xs = nullptr;
        for (int j = 0; j < n_res; j++) {
            int rb_idx = i * n_res + j;
            int rb_kernel = config_.resblock_kernel_sizes[j];
            ggml_tensor* rb_out = build_resblock(gctx, x, weights_.resblocks[rb_idx], rb_kernel);
            capture("vocoder_resblock_" + std::to_string(i) + "_" + std::to_string(j), rb_out);
            if (j == 0) {
                xs = rb_out;
            } else {
                xs = ggml_add(gctx, xs, rb_out);
            }
        }
        x = ggml_scale(gctx, xs, 1.0f / n_res);
        capture("vocoder_resblock_avg_" + std::to_string(i), x);
    }

    // ── Post ────────────────────────────────────────────────────────
    x = snake(gctx, x, weights_.post_act_alpha);
    x = conv1d_vocoder(gctx, weights_.conv_post_w, x, 7, 1, 3, 1, quant_conv1d_ops_);
    x = add_channel_bias(gctx, x, weights_.conv_post_b);
    capture("vocoder_conv_post", x);
    x = ggml_tanh(gctx, x);
    capture("vocoder_tanh", x);

    // ── Build graph ─────────────────────────────────────────────────
    ggml_cgraph* graph = ggml_new_graph_custom(gctx, 1536, false);
    for (ggml_tensor* out : debug_outputs) {
        ggml_build_forward_expand(graph, out);
    }
    ggml_set_name(x, "audio");
    ggml_set_output(x);
    ggml_build_forward_expand(graph, x);

    return graph;
}

// ═════════════════════════════════════════════════════════════════════════
// Full vocoding
// ═════════════════════════════════════════════════════════════════════════

std::vector<float> VocoderModel::vocode(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    ggml_backend_t backend
) {
    // Create graph context
    size_t gctx_size = 1024 * 1024;
    struct ggml_init_params gparams = {
        .mem_size   = gctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* gctx = ggml_init(gparams);
    mem_trace_rss("vocoder ctx init");

    // Create input tensor
    ggml_tensor* mel_t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_mels, n_frames, 1);

    // Build graph
    quant_conv1d_ops_.clear();
    size_t n_conv1d_ops = 2; // pre + post
    for (const auto& dilations : config_.resblock_dilation_sizes) {
        n_conv1d_ops += config_.upsample_rates.size() * dilations.size() * 2;
    }
    quant_conv1d_ops_.reserve(n_conv1d_ops);
    quant_conv_transpose_ops_.clear();
    quant_conv_transpose_ops_.reserve(config_.upsample_rates.size());
    ggml_cgraph* graph = build_vocoder_graph(gctx, mel_t);
    mem_trace_rss("vocoder graph built");

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);
    mem_trace_graph("vocoder", gctx, allocr);
    mem_trace_top_graph_tensors("vocoder", graph);
    mem_trace_rss("vocoder allocated");

    // Set input
    ggml_backend_tensor_set(mel_t, mel.data(), 0, n_mels * n_frames * sizeof(float));
    mem_trace_rss("vocoder input copied");

    // Compute
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VocoderModel] Graph computation failed\n");
    }
    mem_trace_rss("vocoder computed");

    const std::string dump_dir = debug_dump_dir();
    if (!dump_dir.empty()) {
        ensure_debug_dir(dump_dir);
        std::vector<std::string> dump_names = {"vocoder_conv_pre"};
        for (int i = 0; i < (int)config_.upsample_rates.size(); i++) {
            dump_names.push_back("vocoder_upsample_" + std::to_string(i));
            for (int j = 0; j < (int)config_.resblock_kernel_sizes.size(); j++) {
                dump_names.push_back("vocoder_resblock_" + std::to_string(i) + "_" + std::to_string(j));
            }
            dump_names.push_back("vocoder_resblock_avg_" + std::to_string(i));
        }
        dump_names.push_back("vocoder_conv_post");
        dump_names.push_back("vocoder_tanh");

        for (const std::string& name : dump_names) {
            ggml_tensor* t = ggml_get_tensor(gctx, name.c_str());
            if (!t) continue;
            if (t->type != GGML_TYPE_F32 || t->ne[2] != 1) {
                fprintf(stderr, "[VocoderModel] Skipping debug dump for non-F32 tensor %s\n", name.c_str());
                continue;
            }
            const int T = (int)t->ne[0];
            const int C = (int)t->ne[1];
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
            debug_save_f32(
                dump_dir,
                name,
                debug_transpose_time_channel(data, T, C),
                "[" + std::to_string(C) + "," + std::to_string(T) + "]"
            );
        }
    }

    // Extract audio
    ggml_tensor* audio_t = ggml_get_tensor(gctx, "audio");
    if (!audio_t) {
        fprintf(stderr, "[VocoderModel] Failed to locate named vocoder output\n");
        std::abort();
    }
    std::vector<float> audio(ggml_nelements(audio_t));
    ggml_backend_tensor_get(audio_t, audio.data(), 0, ggml_nbytes(audio_t));
    mem_trace_rss("vocoder audio copied");

    // Cleanup
    ggml_gallocr_free(allocr);
    mem_release_to_os();
    mem_trace_rss("vocoder allocator freed");
    ggml_free(gctx);
    mem_release_to_os();
    mem_trace_rss("vocoder context freed");

    return audio;
}

// ═════════════════════════════════════════════════════════════════════════
// Chunked vocoding (for MCU: overlap-discard strategy)
// ═════════════════════════════════════════════════════════════════════════

void VocoderModel::vocode_streaming(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    int chunk_frames,
    ggml_backend_t backend,
    AudioCallback callback
) {
    if (chunk_frames <= 0 || chunk_frames >= n_frames) {
        // No chunking — vocode everything at once
        auto audio = vocode(mel, n_mels, n_frames, backend);
        callback(audio.data(), audio.size());
        return;
    }

    // Overlap-discard: process chunks with overlap to handle
    // convolution receptive fields at boundaries.
    //
    // The receptive field of the vocoder is determined by the
    // convolutions in the resblocks. For kernel sizes (3, 7, 11)
    // with dilations (1, 3, 5), the total receptive field is:
    //   RF = sum over all layers of (K-1) * dilation
    // This is roughly 100-200 frames.
    //
    // We use an overlap of 50% of the receptive field and discard
    // the edges of each chunk.

    int overlap = 64; // Conservative overlap for this vocoder
#if defined(INFLECT_LOW_MEMORY)
    overlap = 8;
#endif
    if (chunk_frames <= overlap) {
        auto audio = vocode(mel, n_mels, n_frames, backend);
        callback(audio.data(), audio.size());
        return;
    }
    int hop = chunk_frames - overlap;
    int upsample = total_upsample();
    std::vector<float> mel_chunk;
    mel_chunk.reserve(n_mels * chunk_frames);
    std::vector<float> audio_chunk;
    std::vector<float> pending_tail;
    std::vector<float> blended;

    for (int start = 0; start < n_frames; start += hop) {
        int end = std::min(start + chunk_frames, n_frames);
        int chunk_len = end - start;

        // Extract mel chunk
        mel_chunk.resize(n_mels * chunk_len);
        for (int m = 0; m < n_mels; m++) {
            for (int t = 0; t < chunk_len; t++) {
                mel_chunk[m + n_mels * t] = mel[m + n_mels * (start + t)];
            }
        }

        // Vocode chunk
        audio_chunk = vocode(mel_chunk, n_mels, chunk_len, backend);

        // Determine which samples to output (discard overlap region)
        int discard_start = (start > 0) ? overlap * upsample / 2 : 0;
        int discard_end = (end < n_frames) ? overlap * upsample / 2 : 0;
        int out_start = discard_start;
        int out_end = audio_chunk.size() - discard_end;

        if (out_end > out_start) {
            const int out_len = out_end - out_start;
            int blended_len = 0;
            if (!pending_tail.empty()) {
                blended_len = std::min<int>(pending_tail.size(), out_len);
                blended.resize(blended_len);
                for (int i = 0; i < blended_len; i++) {
                    const float t = (float)(i + 1) / (float)(blended_len + 1);
                    blended[i] = pending_tail[i] * (1.0f - t) + audio_chunk[out_start + i] * t;
                }
                callback(blended.data(), blended.size());
            }
            if (out_len > blended_len) {
                callback(audio_chunk.data() + out_start + blended_len, out_len - blended_len);
            }
        } else if (!pending_tail.empty() && end >= n_frames) {
            callback(pending_tail.data(), pending_tail.size());
            pending_tail.clear();
        }

        if (discard_end > 0) {
            pending_tail.assign(audio_chunk.begin() + out_end, audio_chunk.end());
        } else {
            pending_tail.clear();
        }
        if (end >= n_frames) {
            break;
        }
    }
}

} // namespace inflect
