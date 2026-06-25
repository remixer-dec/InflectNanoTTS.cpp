#pragma once

#include "model_config.h"
#include <cstdint>
#include <vector>

namespace inflect {

struct GriffinLimConfig {
    int sample_rate = 24000;
    int n_mels = 80;
    int n_fft = 512;
    int hop_length = 256;
    int win_length = 512;
    int iterations = 8;
    float f_min = 0.0f;
    float f_max = 12000.0f;
    float output_peak = 0.35f;
    float max_output_gain = 8192.0f;
    bool use_esp_dsp_fft = false;
    uint64_t seed = 1234;
};

std::vector<float> griffin_lim_vocode(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    const GriffinLimConfig& config
);

void griffin_lim_vocode_streaming(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    const GriffinLimConfig& config,
    AudioCallback callback
);

} // namespace inflect
