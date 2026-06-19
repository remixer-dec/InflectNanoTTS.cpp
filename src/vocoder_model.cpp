#include "vocoder_model.h"
#include <ggml-cpu.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace inflect {

static ggml_tensor* channel_param_3d(ggml_context* gctx, ggml_tensor* t) {
    return ggml_reshape_3d(gctx, t, 1, t->ne[0], 1);
}

static ggml_tensor* add_channel_bias(ggml_context* gctx, ggml_tensor* x, ggml_tensor* bias) {
    return ggml_add(gctx, x, channel_param_3d(gctx, bias));
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
    const ResBlockWeights& w
) {
    // ResBlock1:
    //   for each (c1, c2, a1, a2) in zip(convs1, convs2, acts1, acts2):
    //     y = a1(x) → c1(y) → a2(y) → c2(y) → x = x + y

    for (int i = 0; i < 3; i++) {
        int K = w.convs1_w[i]->ne[0]; // kernel size
        int dilation = config_.resblock_dilation_sizes[i % config_.resblock_dilation_sizes.size()][i];
        int pad1 = (K * dilation - dilation) / 2;
        int pad2 = (K - 1) / 2; // dilation=1 for convs2

        // a1(x)
        ggml_tensor* y = snake(gctx, x, w.acts1_alpha[i]);
        // c1(y) — dilated conv
        y = ggml_conv_1d(gctx, w.convs1_w[i], y, 1, pad1, dilation);
        y = add_channel_bias(gctx, y, w.convs1_b[i]);

        // a2(y)
        y = snake(gctx, y, w.acts2_alpha[i]);
        // c2(y) — dilation=1 conv
        y = ggml_conv_1d(gctx, w.convs2_w[i], y, 1, pad2, 1);
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

    // ── Conv pre ────────────────────────────────────────────────────
    // GGML conv_1d expects input [T, in_ch, B] and weight [K, in_ch, out_ch]
    // Our mel is [n_mels, n_frames, 1] → need to permute to [n_frames, n_mels, 1]
    ggml_tensor* x = ggml_permute(gctx, mel, 1, 0, 2, 3); // [n_frames, n_mels, 1]
    x = ggml_cont(gctx, x);

    x = ggml_conv_1d(gctx, weights_.conv_pre_w, x, 1, 3, 1); // k=7, pad=3
    x = add_channel_bias(gctx, x, weights_.conv_pre_b);

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
        x = ggml_conv_transpose_1d(gctx, weights_.ups_w[i], x, rate, 0, 1);
        x = crop_time_3d(gctx, x, pad, pad);

        // Residual blocks (sum and average)
        ggml_tensor* xs = nullptr;
        for (int j = 0; j < n_res; j++) {
            int rb_idx = i * n_res + j;
            ggml_tensor* rb_out = build_resblock(gctx, x, weights_.resblocks[rb_idx]);
            if (j == 0) {
                xs = rb_out;
            } else {
                xs = ggml_add(gctx, xs, rb_out);
            }
        }
        x = ggml_scale(gctx, xs, 1.0f / n_res);
    }

    // ── Post ────────────────────────────────────────────────────────
    x = snake(gctx, x, weights_.post_act_alpha);
    x = ggml_conv_1d(gctx, weights_.conv_post_w, x, 1, 3, 1);
    x = add_channel_bias(gctx, x, weights_.conv_post_b);
    x = ggml_tanh(gctx, x);

    // ── Build graph ─────────────────────────────────────────────────
    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, x);
    ggml_set_name(x, "audio");

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
    size_t gctx_size = 32 * 1024 * 1024;
    struct ggml_init_params gparams = {
        .mem_size   = gctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* gctx = ggml_init(gparams);

    // Create input tensor
    ggml_tensor* mel_t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_mels, n_frames, 1);

    // Build graph
    ggml_cgraph* graph = build_vocoder_graph(gctx, mel_t);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);

    // Set input
    ggml_backend_tensor_set(mel_t, mel.data(), 0, n_mels * n_frames * sizeof(float));

    // Compute
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VocoderModel] Graph computation failed\n");
    }

    // Extract audio
    ggml_tensor* audio_t = ggml_get_tensor(gctx, "audio");
    if (!audio_t) {
        fprintf(stderr, "[VocoderModel] Failed to locate named vocoder output\n");
        std::abort();
    }
    std::vector<float> audio(ggml_nelements(audio_t));
    ggml_backend_tensor_get(audio_t, audio.data(), 0, ggml_nbytes(audio_t));

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(gctx);

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
    if (chunk_frames <= overlap) {
        auto audio = vocode(mel, n_mels, n_frames, backend);
        callback(audio.data(), audio.size());
        return;
    }
    int hop = chunk_frames - overlap;
    int upsample = total_upsample();

    for (int start = 0; start < n_frames; start += hop) {
        int end = std::min(start + chunk_frames, n_frames);
        int chunk_len = end - start;

        // Extract mel chunk
        std::vector<float> mel_chunk(n_mels * chunk_len);
        for (int m = 0; m < n_mels; m++) {
            for (int t = 0; t < chunk_len; t++) {
                mel_chunk[m + n_mels * t] = mel[m + n_mels * (start + t)];
            }
        }

        // Vocode chunk
        auto audio_chunk = vocode(mel_chunk, n_mels, chunk_len, backend);

        // Determine which samples to output (discard overlap region)
        int discard_start = (start > 0) ? overlap * upsample / 2 : 0;
        int discard_end = (end < n_frames) ? overlap * upsample / 2 : 0;
        int out_start = discard_start;
        int out_end = audio_chunk.size() - discard_end;

        if (out_end > out_start) {
            callback(audio_chunk.data() + out_start, out_end - out_start);
        }
    }
}

} // namespace inflect
