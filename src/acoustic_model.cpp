#include "acoustic_model.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <cstdlib>

namespace inflect {

static void print_tensor_shape(const char* label, const ggml_tensor* t) {
    fprintf(stderr, "%s %s: [%lld, %lld, %lld, %lld]\n",
            label,
            t && t->name[0] ? t->name : "(unnamed)",
            t ? (long long)t->ne[0] : 0,
            t ? (long long)t->ne[1] : 0,
            t ? (long long)t->ne[2] : 0,
            t ? (long long)t->ne[3] : 0);
}

static float tensor_get_f32(const ggml_tensor* t, int64_t i0, int64_t i1 = 0, int64_t i2 = 0) {
    return *(const float*)((const char*)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2]);
}

static void tensor_set_f32(ggml_tensor* t, float v, int64_t i0, int64_t i1, int64_t i2) {
    *(float*)((char*)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2]) = v;
}

// ── Debug: dump a tensor's first few values and stats ───────────────────
void dump_tensor(const ggml_tensor* t, const char* label) {
    if (!t || !t->data) { fprintf(stderr, "[dump] %s: null\n", label); return; }
    int64_t n = ggml_nelements(t);
    int64_t n0 = t->ne[0], n1 = t->ne[1];
    float mn = 1e30f, mx = -1e30f, sum = 0;
    int64_t k = 0;
    // iterate as flat for stats
    for (int64_t i = 0; i < n && i < 100000; i++) {
        float v = tensor_get_f32(t, i % n0, (i / n0) % (n1 > 0 ? n1 : 1), i / (n0 * (n1 > 0 ? n1 : 1)));
        mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
    }
    fprintf(stderr, "[dump] %s (%ld elements) first5=[", label, (long)n);
    for (int64_t i = 0; i < 5 && i < n; i++) {
        float v = tensor_get_f32(t, i % n0, (i / n0) % (n1 > 0 ? n1 : 1), i / (n0 * (n1 > 0 ? n1 : 1)));
        fprintf(stderr, "%.4f%c", v, i < 4 ? ',' : ']');
    }
    fprintf(stderr, " min=%.4f max=%.4f mean=%.4f\n", mn, mx, sum / std::min(n, (int64_t)100000));
}

// ═════════════════════════════════════════════════════════════════════════
// Utility: require dimensions compatible for broadcasting add
// ═════════════════════════════════════════════════════════════════════════

static void require_repeat_compatible(const char* op, const char* where,
                                       const ggml_tensor* a, const ggml_tensor* b) {
    for (int i = 0; i < 4; i++) {
        if (a->ne[i] != b->ne[i] && b->ne[i] != 1 && a->ne[i] != 1) {
            fprintf(stderr, "[AcousticModel] %s %s: shape mismatch a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]\n",
                    op, where,
                    (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
                    (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
            std::abort();
        }
    }
    // Also check that broadcasting is compatible: larger dim % smaller dim == 0
    for (int i = 0; i < 4; i++) {
        int64_t max_dim = std::max(a->ne[i], b->ne[i]);
        int64_t min_dim = std::min(a->ne[i], b->ne[i]);
        if (max_dim != min_dim && (max_dim % min_dim != 0)) {
            fprintf(stderr, "[AcousticModel] %s %s: broadcast incompatible for dim %d: %lld vs %lld\n",
                    op, where, i, (long long)a->ne[i], (long long)b->ne[i]);
            std::abort();
        }
    }
}
static ggml_tensor* checked_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, const char* where) {
    require_repeat_compatible("add", where, a, b);
    return ggml_add(ctx, a, b);
}

// ═════════════════════════════════════════════════════════════════════════
// Layer helpers
// ═════════════════════════════════════════════════════════════════════════

static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* w, ggml_tensor* b) {
    auto layer_norm_op = [](
        ggml_tensor* dst,
        const ggml_tensor* src,
        const ggml_tensor* weight,
        const ggml_tensor* bias,
        int ith,
        int nth,
        void* userdata
    ) {
        (void)userdata;
        const int64_t H = src->ne[0];
        const int64_t T = src->ne[1];
        const int64_t start = (T * ith) / nth;
        const int64_t end = (T * (ith + 1)) / nth;
        for (int64_t t = start; t < end; t++) {
            float mean = 0.0f;
            for (int64_t h = 0; h < H; h++) {
                mean += tensor_get_f32(src, h, t, 0);
            }
            mean /= (float)H;

            float var = 0.0f;
            for (int64_t h = 0; h < H; h++) {
                const float d = tensor_get_f32(src, h, t, 0) - mean;
                var += d * d;
            }
            const float inv_std = 1.0f / std::sqrt(var / (float)H + 1e-5f);

            for (int64_t h = 0; h < H; h++) {
                const float normalized = (tensor_get_f32(src, h, t, 0) - mean) * inv_std;
                const float v = normalized * tensor_get_f32(weight, h, 0, 0) +
                                tensor_get_f32(bias, h, 0, 0);
                tensor_set_f32(dst, v, h, t, 0);
            }
        }
    };
    return ggml_map_custom3(ctx, x, w, b, layer_norm_op, GGML_N_TASKS_MAX, nullptr);
}

static ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                            ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* mat_w = w;
    if (w->ne[0] == 1 && w->ne[1] == x->ne[0]) {
        // Converted Conv1d(kernel=1) weights are stored as [K, in, out].
        // Reinterpret the K=1 slice as the [in, out] matrix expected by GGML.
        mat_w = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);
    }
    if (mat_w->ne[0] != x->ne[0]) {
        fprintf(stderr, "[AcousticModel] linear shape mismatch\n");
        print_tensor_shape("  w", mat_w);
        print_tensor_shape("  x", x);
        print_tensor_shape("  b", b);
        std::abort();
    }
    x = ggml_mul_mat(ctx, mat_w, x);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

// ═════════════════════════════════════════════════════════════════════════
// Depthwise Conv1d with gated activation
// ═════════════════════════════════════════════════════════════════════════

static void depthwise_gated_op(
    ggml_tensor* dst,
    const ggml_tensor* x,
    const ggml_tensor* weight,
    const ggml_tensor* bias,
    int ith,
    int nth,
    void* userdata
) {
    (void)userdata;

    // x: [H, T, 1]  — channels = ne[0], time = ne[1]
    // weight: [K, H, 2]  — kernel, in_channel, filter+gate
    const int64_t H = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t K = weight->ne[0];
    const int64_t pad = K / 2;

    if (weight->ne[1] != H || weight->ne[2] != 2 || bias->ne[0] != 2 * H) {
        fprintf(stderr, "[AcousticModel] invalid depthwise gated tensor shapes\n");
        print_tensor_shape("  x     ", x);
        print_tensor_shape("  weight", weight);
        print_tensor_shape("  bias  ", bias);
        std::abort();
    }

    const int64_t total = H * T;
    const int64_t start = (total * ith) / nth;
    const int64_t end = (total * (ith + 1)) / nth;

    for (int64_t idx = start; idx < end; idx++) {
        const int64_t h = idx % H;   // channel index 0..H-1
        const int64_t t = idx / H;   // time index 0..T-1

        const int64_t a_oc = h;
        const int64_t b_oc = H + h;
        const int64_t out_per_group = 2;
        const int64_t a_src_h = a_oc / out_per_group;
        const int64_t b_src_h = b_oc / out_per_group;

        float a = tensor_get_f32(bias, a_oc);
        float b = tensor_get_f32(bias, b_oc);

        for (int64_t k = 0; k < K; k++) {
            const int64_t src_t = t + k - pad;
            if (src_t < 0 || src_t >= T) {
                continue;
            }
            a += tensor_get_f32(x, a_src_h, src_t, 0) * tensor_get_f32(weight, k, h, 0);
            b += tensor_get_f32(x, b_src_h, src_t, 0) * tensor_get_f32(weight, k, h, 1);
        }

        const float gate = 1.0f / (1.0f + std::exp(-b));
        tensor_set_f32(dst, a * gate, h, t, 0);
    }
}

static ggml_tensor* depthwise_conv_gated(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    ggml_tensor* bias,
    int kernel_size,
    int padding,
    int dilation
) {
    (void)padding;
    (void)dilation;
    return ggml_map_custom3(ctx, x, weight, bias, depthwise_gated_op, 1, nullptr);
}

// ═════════════════════════════════════════════════════════════════════════
// Conv Block
// ═════════════════════════════════════════════════════════════════════════

ggml_tensor* AcousticModel::build_conv_block(
    ggml_context* ctx,
    ggml_tensor* x,
    const ConvBlockWeights& w,
    int ff_mult,
    ggml_tensor* mask
) {
    (void)mask;
    const int H = config_.hidden;
    const int K = config_.kernel_size;

    // Conv branch: x + point(depthwise_gate(norm1(x)))
    ggml_tensor* residual = x;
    ggml_tensor* y = layer_norm(ctx, x, w.norm1_w, w.norm1_b);
    y = depthwise_conv_gated(ctx, y, w.depth_w, w.depth_b, K, K / 2, 1);
    y = linear(ctx, y, w.point_w, w.point_b);
    x = checked_add(ctx, residual, y, "conv_block conv residual");

    // FFN branch: x + Linear(SiLU(Linear(norm2(x))))
    residual = x;
    y = layer_norm(ctx, x, w.norm2_w, w.norm2_b);
    y = linear(ctx, y, w.ff0_w, w.ff0_b);
    y = ggml_silu(ctx, y);
    y = linear(ctx, y, w.ff3_w, w.ff3_b);
    x = checked_add(ctx, residual, y, "conv_block ffn residual");
    return x;
}

AcousticModel::AcousticModel(const AcousticConfig& config)
    : config_(config), weights_{} {}

AcousticModel::~AcousticModel() = default;

ggml_tensor* AcousticModel::layer_norm(ggml_context* ctx, ggml_tensor* x,
                                       ggml_tensor* w, ggml_tensor* b) {
    return inflect::layer_norm(ctx, x, w, b);
}

ggml_tensor* AcousticModel::linear(ggml_context* ctx, ggml_tensor* x,
                                   ggml_tensor* w, ggml_tensor* b) {
    return inflect::linear(ctx, x, w, b);
}

ggml_tensor* AcousticModel::depthwise_conv_gated(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    ggml_tensor* bias,
    int kernel_size,
    int padding,
    int dilation
) {
    return inflect::depthwise_conv_gated(ctx, x, weight, bias, kernel_size, padding, dilation);
}

// ═════════════════════════════════════════════════════════════════════════
// Bidirectional GRU (direct buffer access for CPU backend)
// ═════════════════════════════════════════════════════════════════════════
struct GruOpData {
    ggml_tensor *w_ih, *w_hh, *b_ih, *b_hh;
    ggml_tensor *w_ih_r, *w_hh_r, *b_ih_r, *b_hh_r;
    int hidden_size;
};

static void gru_op(
    ggml_tensor* dst,
    const ggml_tensor* x,
    int ith,
    int nth,
    void* userdata
) {
    (void)nth;
    if (ith != 0) return;

    const auto* d = static_cast<const GruOpData*>(userdata);
    const int input_size = x->ne[0];
    const int T = x->ne[1];
    const int hs = d->hidden_size;

    auto sigmoid = [](float v) {
        return 1.0f / (1.0f + std::exp(-v));
    };
    auto w2 = [](const ggml_tensor* t, int64_t i, int64_t o) {
        return tensor_get_f32(t, i, o, 0);
    };
    auto b1 = [](const ggml_tensor* t, int64_t i) {
        return tensor_get_f32(t, i, 0, 0);
    };

    auto run_direction = [&](bool reverse,
                             const ggml_tensor* w_ih, const ggml_tensor* w_hh,
                             const ggml_tensor* b_ih, const ggml_tensor* b_hh) {
        std::vector<float> h_prev(hs, 0.0f);
        std::vector<float> h_new(hs, 0.0f);

        for (int step = 0; step < T; step++) {
            const int t = reverse ? (T - 1 - step) : step;
            for (int i = 0; i < hs; i++) {
                float gi[3] = {
                    b1(b_ih, 0 * hs + i),
                    b1(b_ih, 1 * hs + i),
                    b1(b_ih, 2 * hs + i),
                };
                float gh[3] = {
                    b1(b_hh, 0 * hs + i),
                    b1(b_hh, 1 * hs + i),
                    b1(b_hh, 2 * hs + i),
                };

                for (int j = 0; j < input_size; j++) {
                    const float xv = tensor_get_f32(x, j, t, 0);
                    gi[0] += xv * w2(w_ih, j, 0 * hs + i);
                    gi[1] += xv * w2(w_ih, j, 1 * hs + i);
                    gi[2] += xv * w2(w_ih, j, 2 * hs + i);
                }
                for (int j = 0; j < hs; j++) {
                    const float hv = h_prev[j];
                    gh[0] += hv * w2(w_hh, j, 0 * hs + i);
                    gh[1] += hv * w2(w_hh, j, 1 * hs + i);
                    gh[2] += hv * w2(w_hh, j, 2 * hs + i);
                }

                // PyTorch GRU gate order is reset, update, new.
                const float r = sigmoid(gi[0] + gh[0]);
                const float z = sigmoid(gi[1] + gh[1]);
                const float n = std::tanh(gi[2] + r * gh[2]);
                h_new[i] = (1.0f - z) * n + z * h_prev[i];
            }

            for (int i = 0; i < hs; i++) {
                h_prev[i] = h_new[i];
                tensor_set_f32(dst, h_new[i], (reverse ? hs : 0) + i, t, 0);
            }
        }
    };

    run_direction(false, d->w_ih, d->w_hh, d->b_ih, d->b_hh);
    run_direction(true, d->w_ih_r, d->w_hh_r, d->b_ih_r, d->b_hh_r);
}

static ggml_tensor* bidirectional_gru(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* w_ih, ggml_tensor* w_hh,
    ggml_tensor* b_ih, ggml_tensor* b_hh,
    ggml_tensor* w_ih_r, ggml_tensor* w_hh_r,
    ggml_tensor* b_ih_r, ggml_tensor* b_hh_r,
    int hidden_size
) {
    auto* data = new GruOpData{w_ih, w_hh, b_ih, b_hh, w_ih_r, w_hh_r, b_ih_r, b_hh_r, hidden_size};
    return ggml_map_custom1(ctx, x, gru_op, 1, data);
}

ggml_tensor* AcousticModel::bidirectional_gru(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* w_ih, ggml_tensor* w_hh,
    ggml_tensor* b_ih, ggml_tensor* b_hh,
    ggml_tensor* w_ih_r, ggml_tensor* w_hh_r,
    ggml_tensor* b_ih_r, ggml_tensor* b_hh_r,
    int hidden_size
) {
    return inflect::bidirectional_gru(ctx, x, w_ih, w_hh, b_ih, b_hh,
                                      w_ih_r, w_hh_r, b_ih_r, b_hh_r,
                                      hidden_size);
}

// ═════════════════════════════════════════════════════════════════════════
// Graph 1: Encoder
// ═════════════════════════════════════════════════════════════════════════

ggml_cgraph* AcousticModel::build_encoder_graph(
    ggml_context* gctx,
    ggml_tensor* phone_ids,  // [seq_len] int32
    ggml_tensor* tone_ids,   // [seq_len] int32
    ggml_tensor* lang_ids,   // [seq_len] int32
    ggml_tensor* speaker_ids // [1] int32
) {
    const int H = config_.hidden;
    const int T = phone_ids->ne[0];

    // ── Embedding lookup ────────────────────────────────────────────
    // ggml_get_rows returns [embedding_width, n_ids].
    ggml_tensor* phone_emb = ggml_get_rows(gctx, weights_.phone_emb, phone_ids); // [H, T]
    phone_emb = ggml_cont(gctx, phone_emb);

    ggml_tensor* tone_emb = ggml_get_rows(gctx, weights_.tone_emb, tone_ids);
    tone_emb = ggml_cont(gctx, tone_emb);

    ggml_tensor* lang_emb = ggml_get_rows(gctx, weights_.lang_emb, lang_ids);
    lang_emb = ggml_cont(gctx, lang_emb);

    // Speaker embedding
    ggml_tensor* spk_emb = ggml_get_rows(gctx, weights_.speaker_emb, speaker_ids); // [1, speaker_dim]
    spk_emb = linear(gctx, spk_emb, weights_.spk_proj_w, weights_.spk_proj_b);     // [H, 1]
    spk_emb = ggml_reshape_3d(gctx, spk_emb, H, 1, 1);                             // [H, 1, 1]

    // Sum embeddings (no sqrt(H) scaling in Inflect-Nano MicroFastSpeech)
    ggml_tensor* x = checked_add(gctx, checked_add(gctx, phone_emb, tone_emb, "phone + tone"), lang_emb, "embeddings + lang");
    x = checked_add(gctx, x, spk_emb, "embeddings + speaker");

    // Save pointer to embedding sum before encoder blocks modify x.
    ggml_tensor* embed_sum = x;
    ggml_set_name(embed_sum, "embed_sum_internal");

    // ── Encoder blocks ──────────────────────────────────────────────
    std::vector<ggml_tensor*> enc_block_outs;
    enc_block_outs.reserve(config_.encoder_layers);
    for (int i = 0; i < config_.encoder_layers; i++) {
        x = build_conv_block(gctx, x, weights_.enc_blocks[i],
                             config_.encoder_ff_mult, nullptr);
        enc_block_outs.push_back(ggml_cpy(gctx, x, ggml_dup_tensor(gctx, x)));
    }

    // ── Prediction heads ────────────────────────────────────────────
    // All heads read from the same encoded tensor
    // Duration
    ggml_tensor* dur = layer_norm(gctx, x, weights_.dur_norm_w, weights_.dur_norm_b);
    dur = linear(gctx, dur, weights_.dur_l1_w, weights_.dur_l1_b);
    dur = ggml_silu(gctx, dur);
    dur = linear(gctx, dur, weights_.dur_l2_w, weights_.dur_l2_b); // [1, T, 1]

    // Energy
    ggml_tensor* energy = layer_norm(gctx, x, weights_.energy_norm_w, weights_.energy_norm_b);
    energy = linear(gctx, energy, weights_.energy_l1_w, weights_.energy_l1_b);
    energy = ggml_silu(gctx, energy);
    energy = linear(gctx, energy, weights_.energy_l2_w, weights_.energy_l2_b); // [1, T, 1]

    // Bright
    ggml_tensor* bright = layer_norm(gctx, x, weights_.bright_norm_w, weights_.bright_norm_b);
    bright = linear(gctx, bright, weights_.bright_l1_w, weights_.bright_l1_b);
    bright = ggml_silu(gctx, bright);
    bright = linear(gctx, bright, weights_.bright_l2_w, weights_.bright_l2_b); // [1, T, 1]

    // Pitch
    ggml_tensor* pitch = layer_norm(gctx, x, weights_.pitch_norm_w, weights_.pitch_norm_b);
    pitch = linear(gctx, pitch, weights_.pitch_l1_w, weights_.pitch_l1_b);
    pitch = ggml_silu(gctx, pitch);
    pitch = linear(gctx, pitch, weights_.pitch_l2_w, weights_.pitch_l2_b); // [2, T, 1]

    // ── Output copies ───────────────────────────────────────────────
    ggml_tensor* embed_sum_out = ggml_cpy(gctx, embed_sum, ggml_dup_tensor(gctx, embed_sum));
    ggml_tensor* encoded_out = ggml_cpy(gctx, x, ggml_dup_tensor(gctx, x));
    ggml_tensor* dur_out     = ggml_cpy(gctx, dur, ggml_dup_tensor(gctx, dur));
    ggml_tensor* energy_out  = ggml_cpy(gctx, energy, ggml_dup_tensor(gctx, energy));
    ggml_tensor* bright_out  = ggml_cpy(gctx, bright, ggml_dup_tensor(gctx, bright));
    ggml_tensor* pitch_out   = ggml_cpy(gctx, pitch, ggml_dup_tensor(gctx, pitch));

    // ── Build graph ─────────────────────────────────────────────────
    ggml_cgraph* graph = ggml_new_graph(gctx);

    ggml_build_forward_expand(graph, embed_sum_out);
    for (ggml_tensor* block_out : enc_block_outs) {
        ggml_build_forward_expand(graph, block_out);
    }
    ggml_build_forward_expand(graph, encoded_out);
    ggml_build_forward_expand(graph, dur_out);
    ggml_build_forward_expand(graph, energy_out);
    ggml_build_forward_expand(graph, bright_out);
    ggml_build_forward_expand(graph, pitch_out);
    // Store output tensor names for extraction
    ggml_set_name(embed_sum_out, "embed_sum");
    for (int i = 0; i < (int)enc_block_outs.size(); i++) {
        ggml_set_name(enc_block_outs[i], ("enc_block_" + std::to_string(i)).c_str());
    }
    ggml_set_name(encoded_out, "encoded");
    ggml_set_name(dur_out,     "log_durations");
    ggml_set_name(energy_out,  "energy");
    ggml_set_name(bright_out,  "bright");
    ggml_set_name(pitch_out,   "pitch");

    return graph;
}

// ═════════════════════════════════════════════════════════════════════════
// Graph 2: Decoder + Mel Head + Postnet
// ═════════════════════════════════════════════════════════════════════════

ggml_cgraph* AcousticModel::build_decoder_graph(
    ggml_context* gctx,
    ggml_tensor* features  // [H, n_frames, 1]
) {
    const int H = config_.hidden;
    const int n_mels = config_.n_mels;
    const int T = features->ne[1];

    ggml_tensor* x = features;

    // ── Decoder blocks ──────────────────────────────────────────────
    for (int i = 0; i < config_.decoder_layers; i++) {
        x = build_conv_block(gctx, x, weights_.dec_blocks[i],
                             config_.decoder_ff_mult, nullptr);
    }

    // ── Bidirectional GRU ───────────────────────────────────────────
    int gru_hidden = H / 2; // 84, concatenated to 168
    ggml_tensor* gru = bidirectional_gru(gctx, x,
        weights_.gru_w_ih, weights_.gru_w_hh,
        weights_.gru_b_ih, weights_.gru_b_hh,
        weights_.gru_w_ih_r, weights_.gru_w_hh_r,
        weights_.gru_b_ih_r, weights_.gru_b_hh_r,
        gru_hidden); // [H, T, 1]
    x = checked_add(gctx, x, gru, "frame_gru residual");

    // ── Mel head ────────────────────────────────────────────────────
    x = layer_norm(gctx, x, weights_.mel_norm_w, weights_.mel_norm_b);
    x = linear(gctx, x, weights_.mel_l1_w, weights_.mel_l1_b);
    x = ggml_silu(gctx, x);
    x = linear(gctx, x, weights_.mel_l2_w, weights_.mel_l2_b); // [n_mels, T, 1]
    ggml_tensor* mel = x;

    // ── Postnet ─────────────────────────────────────────────────────
    // Postnet runs in GGML conv layout [T, C, B], then returns to [C, T, B].
    ggml_tensor* post = ggml_permute(gctx, mel, 1, 0, 2, 3);
    post = ggml_cont(gctx, post);
    post = ggml_conv_1d(gctx, ggml_cast(gctx, weights_.post0_w, GGML_TYPE_F16), post, 1, 2, 1);
    post = checked_add(gctx, post, ggml_reshape_3d(gctx, weights_.post0_b, 1, H, 1), "postnet.0 bias");
    post = ggml_tanh(gctx, post);
    post = ggml_conv_1d(gctx, ggml_cast(gctx, weights_.post2_w, GGML_TYPE_F16), post, 1, 2, 1);
    post = checked_add(gctx, post, ggml_reshape_3d(gctx, weights_.post2_b, 1, H, 1), "postnet.2 bias");
    post = ggml_tanh(gctx, post);
    post = ggml_conv_1d(gctx, ggml_cast(gctx, weights_.post4_w, GGML_TYPE_F16), post, 1, 2, 1);
    post = checked_add(gctx, post, ggml_reshape_3d(gctx, weights_.post4_b, 1, n_mels, 1), "postnet.4 bias");
    post = ggml_permute(gctx, post, 1, 0, 2, 3);
    post = ggml_cont(gctx, post);
    // Scale postnet output and add to mel
    post = ggml_scale(gctx, post, 0.1f);
    x = checked_add(gctx, mel, post, "postnet residual");

    // ── Build graph ─────────────────────────────────────────────────
    ggml_set_name(x, "mel");
    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, x);
    return graph;
}

// ═════════════════════════════════════════════════════════════════════════
// Graph Execution: Encoder
// ═════════════════════════════════════════════════════════════════════════

EncoderOutput AcousticModel::run_encoder(
    const std::vector<int32_t>& phone_ids,
    const std::vector<int32_t>& tone_ids,
    const std::vector<int32_t>& lang_ids,
    int speaker_id,
    ggml_backend_t backend
) {
    const int T = phone_ids.size();
    const int H = config_.hidden;

    // Create graph context
    size_t gctx_size = 16 * 1024 * 1024;
    struct ggml_init_params gparams = {
        .mem_size   = gctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* gctx = ggml_init(gparams);

    // Create input tensors (these live in the backend buffer)
    ggml_tensor* phone_t  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_tensor* tone_t   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_tensor* lang_t   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_tensor* speaker_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);

    int32_t sid = speaker_id;

    // Build graph
    ggml_cgraph* graph = build_encoder_graph(gctx, phone_t, tone_t, lang_t, speaker_t);

    // Allocate and compute
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);

    // Set input tensors after allocation
    ggml_backend_tensor_set(phone_t, phone_ids.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(tone_t, tone_ids.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(lang_t, lang_ids.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(speaker_t, &sid, 0, sizeof(int32_t));

    // Compute
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[AcousticModel] Graph computation failed\n");
    }

    // Extract outputs
    EncoderOutput out;
    out.seq_len = T;
    out.hidden = H;

    auto to_host = [](ggml_tensor* t) -> std::vector<float> {
        std::vector<float> data(ggml_nelements(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        return data;
    };

    // Read embed_sum FIRST (it is marked OUTPUT so the allocator keeps its buffer)
    ggml_tensor* esum_t = ggml_get_tensor(gctx, "embed_sum");
    if (esum_t) {
        out.embed_sum = to_host(esum_t);
    }

    out.enc_blocks.clear();
    out.enc_blocks.reserve(config_.encoder_layers);
    for (int i = 0; i < config_.encoder_layers; i++) {
        ggml_tensor* block_t = ggml_get_tensor(gctx, ("enc_block_" + std::to_string(i)).c_str());
        if (block_t) {
            out.enc_blocks.push_back(to_host(block_t));
        }
    }

    ggml_tensor* encoded_t = ggml_get_tensor(gctx, "encoded");
    ggml_tensor* dur_t     = ggml_get_tensor(gctx, "log_durations");
    ggml_tensor* energy_t  = ggml_get_tensor(gctx, "energy");
    ggml_tensor* bright_t  = ggml_get_tensor(gctx, "bright");
    ggml_tensor* pitch_t   = ggml_get_tensor(gctx, "pitch");
    if (!encoded_t || !dur_t || !energy_t || !bright_t || !pitch_t) {
        fprintf(stderr, "[AcousticModel] Failed to locate named encoder outputs\n");
        std::abort();
    }

    out.encoded       = to_host(encoded_t);
    out.log_durations = to_host(dur_t);
    out.energy        = to_host(energy_t);
    out.bright        = to_host(bright_t);
    out.pitch         = to_host(pitch_t);

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(gctx);

    return out;
}

// ═════════════════════════════════════════════════════════════════════════
// CPU Bridge: Length Regulation
//
// This expands phoneme-level features to frame-level based on predicted
// durations. Also adds prosody projections and absolute frame positional
// embeddings.
// ═════════════════════════════════════════════════════════════════════════

RegulatedFeatures AcousticModel::length_regulate(
    const EncoderOutput& enc,
    float length_scale,
    float pitch_scale,
    float energy_scale
) {
    const int H = config_.hidden;
    const int T = enc.seq_len;

    auto silu = [](float x) -> float {
        return x / (1.0f + std::exp(-x));
    };

    const float* dur_data    = enc.log_durations.data();
    const float* energy_pred = enc.energy.data();
    const float* bright_pred = enc.bright.data();
    const float* pitch_pred  = enc.pitch.data(); // GGML layout: component + token * 2

    auto weight = [](const ggml_tensor* t, int64_t in, int64_t out) -> float {
        return tensor_get_f32(t, in, out, 0);
    };
    auto bias = [](const ggml_tensor* t, int64_t out) -> float {
        return tensor_get_f32(t, out, 0, 0);
    };
    auto linear_full = [&](const float* input, int in_dim,
                           const ggml_tensor* w, const ggml_tensor* b,
                           float* output, int out_dim) {
        for (int o = 0; o < out_dim; o++) {
            float v = bias(b, o);
            for (int i = 0; i < in_dim; i++) {
                v += input[i] * weight(w, i, o);
            }
            output[o] = v;
        }
    };

    // ── Compute phoneme-level durations ─────────────────────────────
    // Apply expm1 with clamp, multiply by length_scale, round, clamp to at least 1
    std::vector<int32_t> durations(T);
    int total_frames = 0;
    for (int i = 0; i < T; i++) {
        float d = std::expm1(dur_data[i]);
        d = std::clamp(d, 0.0f, 80.0f);
        d *= length_scale;
        int di = (int)std::round(d);
        di = std::max(1, di);
        durations[i] = (int32_t)di;
        total_frames += di;
    }
    total_frames = std::min(total_frames, config_.max_frames);

    // ── Token-level conditioning: encoded + energy_proj + bright_proj ──
    std::vector<float> conditioned(T * H);
    for (int t = 0; t < T; t++) {
        const float energy = energy_pred[t] * energy_scale;
        const float bright = bright_pred[t];
        for (int h = 0; h < H; h++) {
            float v = enc.encoded[h + t * H];
            v += energy * weight(weights_.energy_proj_w, 0, h) + bias(weights_.energy_proj_b, h);
            v += bright * weight(weights_.bright_proj_w, 0, h) + bias(weights_.bright_proj_b, h);
            conditioned[t * H + h] = v;
        }
    }

    // ── Expand to frame-level and add frame/prosody/context features ──
    int n_frames = total_frames;
    std::vector<float> regulated(n_frames * H);
    std::vector<float> frame_hidden(H);
    std::vector<float> frame_proj(H);
    std::vector<float> pitch_hidden(H);
    std::vector<float> pitch_proj(H);
    std::vector<float> ctx_in(3 * H);
    std::vector<float> ctx_hidden(2 * H);
    std::vector<float> ctx_proj(H);

    const int token_count = std::max(1, T);
    int f = 0;
    for (int t = 0; t < T && f < n_frames; t++) {
        const int dur = durations[t];
        const int prev_t = std::max(0, t - 1);
        const int next_t = std::min(T - 1, t + 1);

        for (int h = 0; h < H; h++) {
            ctx_in[h] = conditioned[prev_t * H + h];
            ctx_in[H + h] = conditioned[t * H + h];
            ctx_in[2 * H + h] = conditioned[next_t * H + h];
        }
        linear_full(ctx_in.data(), 3 * H, weights_.lctx0_w, weights_.lctx0_b,
                    ctx_hidden.data(), 2 * H);
        for (float& v : ctx_hidden) v = silu(v);
        linear_full(ctx_hidden.data(), 2 * H, weights_.lctx2_w, weights_.lctx2_b,
                    ctx_proj.data(), H);

        float pitch_in[2] = {
            pitch_pred[0 + t * 2] * pitch_scale,
            std::clamp(pitch_pred[1 + t * 2], 0.0f, 1.0f),
        };
        linear_full(pitch_in, 2, weights_.pitch_proj0_w, weights_.pitch_proj0_b,
                    pitch_hidden.data(), H);
        for (float& v : pitch_hidden) v = silu(v);
        linear_full(pitch_hidden.data(), H, weights_.pitch_proj2_w, weights_.pitch_proj2_b,
                    pitch_proj.data(), H);

        for (int r = 0; r < dur && f < n_frames; r++, f++) {
            const float rel = dur > 1 ? (float)r / (float)(dur - 1) : 0.0f;
            const float meta[8] = {
                rel,
                1.0f - rel,
                1.0f - std::fabs(rel * 2.0f - 1.0f),
                std::sin(rel * (float)M_PI),
                std::cos(rel * (float)M_PI),
                (float)t / (float)std::max(1, token_count - 1),
                std::log1p((float)dur) / 6.0f,
                (float)dur / 40.0f,
            };
            linear_full(meta, 8, weights_.frame_proj0_w, weights_.frame_proj0_b,
                        frame_hidden.data(), H);
            for (float& v : frame_hidden) v = silu(v);
            linear_full(frame_hidden.data(), H, weights_.frame_proj2_w, weights_.frame_proj2_b,
                        frame_proj.data(), H);

            int bin = (f * config_.abs_frame_bins) / std::max(1, config_.max_frames);
            bin = std::clamp(bin, 0, config_.abs_frame_bins - 1);

            for (int h = 0; h < H; h++) {
                regulated[f * H + h] =
                    conditioned[t * H + h] +
                    frame_proj[h] +
                    ctx_proj[h] +
                    tensor_get_f32(weights_.abs_frame_emb, h, bin, 0) +
                    pitch_proj[h];
            }
        }
    }

    return {regulated, durations, n_frames, H};
}

// ═════════════════════════════════════════════════════════════════════════
// Graph Execution: Decoder
// ═════════════════════════════════════════════════════════════════════════

std::vector<float> AcousticModel::run_decoder(
    const RegulatedFeatures& features,
    ggml_backend_t backend
) {
    const int H = config_.hidden;
    const int n_mels = config_.n_mels;

    // Create graph context + input tensor
    size_t gctx_size = 16 * 1024 * 1024;
    struct ggml_init_params gparams = {
        .mem_size   = gctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* gctx = ggml_init(gparams);

    // Create FEATURES input tensor
    ggml_tensor* features_t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, H, features.n_frames, 1);
    ggml_set_input(features_t);

    // Build graph
    ggml_cgraph* graph = build_decoder_graph(gctx, features_t);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);

    // Set input
    ggml_backend_tensor_set(features_t, features.features.data(), 0, features.features.size() * sizeof(float));

    // Compute
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[AcousticModel] Decoder graph compute failed\n");
    }

    // Extract mel
    ggml_tensor* mel_t = ggml_get_tensor(gctx, "mel");
    if (!mel_t) {
        fprintf(stderr, "[AcousticModel] mel tensor not found\n");
        ggml_gallocr_free(allocr);
        ggml_free(gctx);
        return {};
    }

    std::vector<float> mel(ggml_nelements(mel_t));
    ggml_backend_tensor_get(mel_t, mel.data(), 0, ggml_nbytes(mel_t));

    ggml_gallocr_free(allocr);
    ggml_free(gctx);

    return mel;
}

// ═════════════════════════════════════════════════════════════════════════
// Weights Loading
// ═════════════════════════════════════════════════════════════════════════

void AcousticModel::load_conv_block(ModelLoader& loader, ConvBlockWeights& blk,
                                     const std::string& prefix) {
    blk.norm1_w  = loader.get_tensor(prefix + ".norm1.weight");
    blk.norm1_b  = loader.get_tensor(prefix + ".norm1.bias");
    blk.depth_w  = loader.get_tensor(prefix + ".depth.weight");
    blk.depth_b  = loader.get_tensor(prefix + ".depth.bias");
    blk.point_w  = loader.get_tensor(prefix + ".point.weight");
    blk.point_b  = loader.get_tensor(prefix + ".point.bias");
    blk.norm2_w  = loader.get_tensor(prefix + ".norm2.weight");
    blk.norm2_b  = loader.get_tensor(prefix + ".norm2.bias");
    blk.ff0_w    = loader.get_tensor(prefix + ".ff.0.weight");
    blk.ff0_b    = loader.get_tensor(prefix + ".ff.0.bias");
    blk.ff3_w    = loader.get_tensor(prefix + ".ff.3.weight");
    blk.ff3_b    = loader.get_tensor(prefix + ".ff.3.bias");
}

bool AcousticModel::load(ModelLoader& loader) {
    // Embeddings
    weights_.phone_emb   = loader.get_tensor("phone.weight");
    weights_.tone_emb    = loader.get_tensor("tone.weight");
    weights_.lang_emb    = loader.get_tensor("lang.weight");
    weights_.speaker_emb = loader.get_tensor("speaker.weight");
    weights_.spk_proj_w  = loader.get_tensor("speaker_proj.weight");
    weights_.spk_proj_b  = loader.get_tensor("speaker_proj.bias");

    // Encoder blocks
    for (int i = 0; i < config_.encoder_layers; i++) {
        ConvBlockWeights blk;
        load_conv_block(loader, blk, "encoder." + std::to_string(i));
        weights_.enc_blocks.push_back(blk);
    }

    // Duration head
    weights_.dur_norm_w = loader.get_tensor("duration_head.0.weight");
    weights_.dur_norm_b = loader.get_tensor("duration_head.0.bias");
    weights_.dur_l1_w   = loader.get_tensor("duration_head.1.weight");
    weights_.dur_l1_b   = loader.get_tensor("duration_head.1.bias");
    weights_.dur_l2_w   = loader.get_tensor("duration_head.3.weight");
    weights_.dur_l2_b   = loader.get_tensor("duration_head.3.bias");

    // Energy head
    weights_.energy_norm_w = loader.get_tensor("energy_head.0.weight");
    weights_.energy_norm_b = loader.get_tensor("energy_head.0.bias");
    weights_.energy_l1_w   = loader.get_tensor("energy_head.1.weight");
    weights_.energy_l1_b   = loader.get_tensor("energy_head.1.bias");
    weights_.energy_l2_w   = loader.get_tensor("energy_head.3.weight");
    weights_.energy_l2_b   = loader.get_tensor("energy_head.3.bias");

    // Bright head
    weights_.bright_norm_w = loader.get_tensor("bright_head.0.weight");
    weights_.bright_norm_b = loader.get_tensor("bright_head.0.bias");
    weights_.bright_l1_w   = loader.get_tensor("bright_head.1.weight");
    weights_.bright_l1_b   = loader.get_tensor("bright_head.1.bias");
    weights_.bright_l2_w   = loader.get_tensor("bright_head.3.weight");
    weights_.bright_l2_b   = loader.get_tensor("bright_head.3.bias");

    // Pitch head
    weights_.pitch_norm_w = loader.get_tensor("pitch_head.0.weight");
    weights_.pitch_norm_b = loader.get_tensor("pitch_head.0.bias");
    weights_.pitch_l1_w   = loader.get_tensor("pitch_head.1.weight");
    weights_.pitch_l1_b   = loader.get_tensor("pitch_head.1.bias");
    weights_.pitch_l2_w   = loader.get_tensor("pitch_head.3.weight");
    weights_.pitch_l2_b   = loader.get_tensor("pitch_head.3.bias");

    // Prosody projections
    weights_.energy_proj_w  = loader.get_tensor("energy_proj.weight");
    weights_.energy_proj_b  = loader.get_tensor("energy_proj.bias");
    weights_.bright_proj_w  = loader.get_tensor("bright_proj.weight");
    weights_.bright_proj_b  = loader.get_tensor("bright_proj.bias");
    weights_.pitch_proj0_w  = loader.get_tensor("pitch_proj.0.weight");
    weights_.pitch_proj0_b  = loader.get_tensor("pitch_proj.0.bias");
    weights_.pitch_proj2_w  = loader.get_tensor("pitch_proj.2.weight");
    weights_.pitch_proj2_b  = loader.get_tensor("pitch_proj.2.bias");

    // Frame-level features
    weights_.abs_frame_emb  = loader.get_tensor("abs_frame.weight");
    weights_.frame_proj0_w  = loader.get_tensor("frame_proj.0.weight");
    weights_.frame_proj0_b  = loader.get_tensor("frame_proj.0.bias");
    weights_.frame_proj2_w  = loader.get_tensor("frame_proj.2.weight");
    weights_.frame_proj2_b  = loader.get_tensor("frame_proj.2.bias");

    // Local context
    weights_.lctx0_w = loader.get_tensor("local_ctx.0.weight");
    weights_.lctx0_b = loader.get_tensor("local_ctx.0.bias");
    weights_.lctx2_w = loader.get_tensor("local_ctx.2.weight");
    weights_.lctx2_b = loader.get_tensor("local_ctx.2.bias");

    // Decoder blocks
    for (int i = 0; i < config_.decoder_layers; i++) {
        ConvBlockWeights blk;
        load_conv_block(loader, blk, "decoder." + std::to_string(i));
        weights_.dec_blocks.push_back(blk);
    }

    // Bidirectional GRU
    weights_.gru_w_ih   = loader.get_tensor("frame_gru.weight_ih_l0");
    weights_.gru_w_hh   = loader.get_tensor("frame_gru.weight_hh_l0");
    weights_.gru_b_ih   = loader.get_tensor("frame_gru.bias_ih_l0");
    weights_.gru_b_hh   = loader.get_tensor("frame_gru.bias_hh_l0");
    weights_.gru_w_ih_r = loader.get_tensor("frame_gru.weight_ih_l0_reverse");
    weights_.gru_w_hh_r = loader.get_tensor("frame_gru.weight_hh_l0_reverse");
    weights_.gru_b_ih_r = loader.get_tensor("frame_gru.bias_ih_l0_reverse");
    weights_.gru_b_hh_r = loader.get_tensor("frame_gru.bias_hh_l0_reverse");

    // Mel head
    weights_.mel_norm_w = loader.get_tensor("mel_head.0.weight");
    weights_.mel_norm_b = loader.get_tensor("mel_head.0.bias");
    weights_.mel_l1_w   = loader.get_tensor("mel_head.1.weight");
    weights_.mel_l1_b   = loader.get_tensor("mel_head.1.bias");
    weights_.mel_l2_w   = loader.get_tensor("mel_head.3.weight");
    weights_.mel_l2_b   = loader.get_tensor("mel_head.3.bias");

    // Postnet
    weights_.post0_w = loader.get_tensor("postnet.0.weight");
    weights_.post0_b = loader.get_tensor("postnet.0.bias");
    weights_.post2_w = loader.get_tensor("postnet.2.weight");
    weights_.post2_b = loader.get_tensor("postnet.2.bias");
    weights_.post4_w = loader.get_tensor("postnet.4.weight");
    weights_.post4_b = loader.get_tensor("postnet.4.bias");

    wctx_ = loader.ctx();
    return true;
}

} // namespace inflect
