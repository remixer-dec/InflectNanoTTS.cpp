#pragma once

#include "model_config.h"
#include "model_loader.h"
#include <ggml.h>
#include <ggml-backend.h>
#include <vector>
#include <memory>

namespace inflect {

// ─────────────────────────────────────────────────────────────────────────
// Weight containers
// ─────────────────────────────────────────────────────────────────────────

// Each encoder/decoder block:
//   x + point(depthwise_gate(norm1(x)))
//   then x + ff(norm2(x))
struct ConvBlockWeights {
    ggml_tensor *norm1_w, *norm1_b;
    ggml_tensor *depth_w, *depth_b;   // depthwise conv: PyTorch [2H, 1, K]
    ggml_tensor *point_w, *point_b;   // pointwise conv: PyTorch [H, H, 1]
    ggml_tensor *norm2_w, *norm2_b;
    ggml_tensor *ff0_w, *ff0_b;       // Linear(H, ff_mult*H)
    ggml_tensor *ff3_w, *ff3_b;       // Linear(ff_mult*H, H)
};

struct AcousticWeights {
    // Embeddings
    ggml_tensor *phone_emb;    // [vocab, hidden]
    ggml_tensor *tone_emb;     // [tone_size, hidden]
    ggml_tensor *lang_emb;     // [lang_size, hidden]
    ggml_tensor *speaker_emb;  // [speaker_count, speaker_dim]
    ggml_tensor *spk_proj_w, *spk_proj_b; // [hidden, speaker_dim], [hidden]

    // Encoder blocks (5)
    std::vector<ConvBlockWeights> enc_blocks;

    // Duration head: LayerNorm → Linear(H,H) → ReLU → Linear(H,1)
    ggml_tensor *dur_norm_w, *dur_norm_b;
    ggml_tensor *dur_l1_w, *dur_l1_b;
    ggml_tensor *dur_l2_w, *dur_l2_b;

    // Energy head: LayerNorm → Linear(H//2, H) → ReLU → Linear(H//2, 1)
    ggml_tensor *energy_norm_w, *energy_norm_b;
    ggml_tensor *energy_l1_w, *energy_l1_b;
    ggml_tensor *energy_l2_w, *energy_l2_b;

    // Bright head (same shape as energy)
    ggml_tensor *bright_norm_w, *bright_norm_b;
    ggml_tensor *bright_l1_w, *bright_l1_b;
    ggml_tensor *bright_l2_w, *bright_l2_b;

    // Pitch head: LayerNorm → Linear(H, H) → ReLU → Linear(H, 2)
    ggml_tensor *pitch_norm_w, *pitch_norm_b;
    ggml_tensor *pitch_l1_w, *pitch_l1_b;
    ggml_tensor *pitch_l2_w, *pitch_l2_b;

    // Prosody projections (applied after length regulation)
    ggml_tensor *energy_proj_w, *energy_proj_b; // [H, 1], [H]
    ggml_tensor *bright_proj_w, *bright_proj_b; // [H, 1], [H]
    ggml_tensor *pitch_proj0_w, *pitch_proj0_b; // [H, 2], [H]
    ggml_tensor *pitch_proj2_w, *pitch_proj2_b; // [H, H], [H]

    // Frame-level features
    ggml_tensor *abs_frame_emb;  // [abs_frame_bins, hidden]
    ggml_tensor *frame_proj0_w, *frame_proj0_b; // [H, 8], [H]
    ggml_tensor *frame_proj2_w, *frame_proj2_b; // [H, H], [H]

    // Local context conv (2-layer)
    ggml_tensor *lctx0_w, *lctx0_b; // [2H, 3H], [2H]  (3H = concat of 3 features)
    ggml_tensor *lctx2_w, *lctx2_b; // [H, 2H], [H]

    // Decoder blocks (6)
    std::vector<ConvBlockWeights> dec_blocks;

    // Bidirectional GRU
    ggml_tensor *gru_w_ih, *gru_w_hh, *gru_b_ih, *gru_b_hh;
    ggml_tensor *gru_w_ih_r, *gru_w_hh_r, *gru_b_ih_r, *gru_b_hh_r;

    // Mel head: LayerNorm → Linear(H,H) → ReLU → Linear(n_mels, H)
    ggml_tensor *mel_norm_w, *mel_norm_b;
    ggml_tensor *mel_l1_w, *mel_l1_b;
    ggml_tensor *mel_l2_w, *mel_l2_b;

    // Postnet: Conv1d(80→H, k=5) → Conv1d(H→H, k=5) → Conv1d(H→80, k=5)
    ggml_tensor *post0_w, *post0_b;
    ggml_tensor *post2_w, *post2_b;
    ggml_tensor *post4_w, *post4_b;
};

struct QuantConv1dOpData {
    int kernel_size;
    int stride;
    int padding;
    int dilation;
};

// ─────────────────────────────────────────────────────────────────────────
// Intermediate results
// ─────────────────────────────────────────────────────────────────────────

struct EncoderOutput {
    std::vector<float> encoded;       // [hidden, seq_len] (row-major: hidden rows, seq_len cols)
    std::vector<float> embed_sum;     // [hidden, seq_len] embedding sum before encoder blocks
    std::vector<std::vector<float>> enc_blocks; // per-block outputs, same shape as encoded
    std::vector<float> log_durations; // [seq_len]
    std::vector<float> energy;        // [seq_len]
    std::vector<float> bright;        // [seq_len]
    std::vector<float> pitch;         // [2, seq_len]
    int seq_len;
    int hidden;
};

struct RegulatedFeatures {
    std::vector<float> features; // [hidden, n_frames]
    std::vector<int32_t> durations; // [seq_len]
    int n_frames;
    int hidden;
};

// ─────────────────────────────────────────────────────────────────────────
// Acoustic model
// ─────────────────────────────────────────────────────────────────────────

class AcousticModel {
public:
    AcousticModel(const AcousticConfig& config);
    ~AcousticModel();

    bool load(ModelLoader& loader);

    // Graph 1: Encoder + prediction heads
    EncoderOutput run_encoder(
        const std::vector<int32_t>& phone_ids,
        const std::vector<int32_t>& tone_ids,
        const std::vector<int32_t>& lang_ids,
        int speaker_id,
        ggml_backend_t backend
    );

    // CPU bridge: length regulation (expand by durations, add prosody + positional)
    RegulatedFeatures length_regulate(
        const EncoderOutput& enc,
        float length_scale,
        float pitch_scale,
        float energy_scale
    );

    // Graph 2: Decoder + mel head + postnet
    std::vector<float> run_decoder(
        const RegulatedFeatures& features,
        ggml_backend_t backend
    );

    const AcousticConfig& config() const { return config_; }

private:
    AcousticConfig config_;
    AcousticWeights weights_;
    ggml_context* wctx_ = nullptr;  // holds weight tensor metadata
    std::vector<QuantConv1dOpData> quant_conv1d_ops_;

    // ── Graph builders ─────────────────────────────────────────────
    ggml_cgraph* build_encoder_graph(
        ggml_context* gctx,
        ggml_tensor* phone_ids,
        ggml_tensor* tone_ids,
        ggml_tensor* lang_ids,
        ggml_tensor* speaker_ids
    );

    ggml_cgraph* build_decoder_graph(
        ggml_context* gctx,
        ggml_tensor* features,  // [n_frames, hidden, 1]
        void* gru_op_data
    );

    // ── Layer helpers ──────────────────────────────────────────────
    ggml_tensor* build_conv_block(
        ggml_context* gctx,
        ggml_tensor* x,
        const ConvBlockWeights& w,
        int ff_mult,
        ggml_tensor* mask = nullptr
    );

    ggml_tensor* layer_norm(ggml_context* gctx, ggml_tensor* x,
                            ggml_tensor* w, ggml_tensor* b);

    ggml_tensor* linear(ggml_context* gctx, ggml_tensor* x,
                        ggml_tensor* w, ggml_tensor* b);

    // Depthwise conv1d with gated activation (folded into one op)
    // Input:  [T, C, 1]
    // Weight: [K, 1, 2*C]  (PyTorch order, transposed to GGML at convert time)
    // Output: [T_out, C, 1] (after tanh*sigmoid gating)
    ggml_tensor* depthwise_conv_gated(
        ggml_context* gctx,
        ggml_tensor* x,
        ggml_tensor* weight,
        ggml_tensor* bias,
        int kernel_size,
        int padding,
        int dilation
    );

    // ── Utility ────────────────────────────────────────────────────
    void load_conv_block(ModelLoader& loader, ConvBlockWeights& blk,
                         const std::string& prefix);
};

} // namespace inflect
