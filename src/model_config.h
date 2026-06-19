#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>

namespace inflect {

// ─────────────────────────────────────────────────────────────────────────
// Acoustic Model Configuration (FastSpeech2-style)
// ─────────────────────────────────────────────────────────────────────────

struct AcousticConfig {
    int vocab_size       = 256;
    int tone_size        = 16;
    int lang_size        = 4;
    int n_mels           = 80;
    int hidden           = 168;
    int encoder_layers   = 5;
    int decoder_layers   = 6;
    int encoder_ff_mult  = 4;   // 672 / 168
    int decoder_ff_mult  = 3;   // 504 / 168
    int kernel_size      = 7;
    int speaker_count    = 2;
    int speaker_dim      = 64;
    int abs_frame_bins   = 512;
    int sample_rate      = 24000;
    int max_frames       = 1400;
    float postnet_scale  = 0.1f;
    bool use_frame_pitch = true;
};

// ─────────────────────────────────────────────────────────────────────────
// Vocoder Configuration (HiFi-GAN)
// ─────────────────────────────────────────────────────────────────────────

struct VocoderConfig {
    int sample_rate              = 24000;
    int num_mels                 = 80;
    int upsample_initial_channel = 144;
    std::vector<int> upsample_rates       = {8, 8, 2, 2};
    std::vector<int> upsample_kernel_sizes = {16, 16, 4, 4};
    std::vector<int> resblock_kernel_sizes = {3, 7, 11};
    // Default dilations for ResBlock1
    std::vector<std::vector<int>> resblock_dilation_sizes = {
        {1, 3, 5}, {1, 3, 5}, {1, 3, 5}
    };
    std::string activation = "snake";
};

// ─────────────────────────────────────────────────────────────────────────
// Synthesis Parameters
// ─────────────────────────────────────────────────────────────────────────

struct SynthParams {
    float length_scale = 1.0f;
    float pitch_scale  = 1.0f;
    float energy_scale = 1.0f;
    int   speaker_id   = 0;
    uint64_t seed      = 1234;
};

// ─────────────────────────────────────────────────────────────────────────
// Text Frontend Structures
// ─────────────────────────────────────────────────────────────────────────

struct TokenSequence {
    std::vector<int32_t> phone_ids;
    std::vector<int32_t> tone_ids;
    std::vector<int32_t> lang_ids;
};

// ─────────────────────────────────────────────────────────────────────────
// Streaming Audio Callback
// ─────────────────────────────────────────────────────────────────────────

// Called by the vocoder when a chunk of audio is ready.
// samples: pointer to the audio data
// n_samples: number of floats in the buffer
using AudioCallback = std::function<void(const float* samples, size_t n_samples)>;

} // namespace inflect
