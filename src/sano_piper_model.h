#pragma once

#include "sano_config.h"
#include "sano_piper_decoder.h"

#include <cstdint>
#include <cmath>
#include <vector>

namespace inflect {

inline int32_t sano_round_half_even(float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    if (fraction < 0.5f) return static_cast<int32_t>(lower);
    if (fraction > 0.5f) return static_cast<int32_t>(lower + 1.0f);
    const int32_t lower_int = static_cast<int32_t>(lower);
    return (lower_int & 1) == 0 ? lower_int : lower_int + 1;
}

struct SanoPiperFrontOutput {
    std::vector<int32_t> durations;
    std::vector<float> latent; // GGML/channel-major [channels][frames]
    int frames = 0;
    int channels = 0;
};

class SanoPiperModel {
public:
    SanoPiperModel() = default;

    bool configure(ModelLoader& loader);

    std::vector<int32_t> predict_durations(
        ModelLoader& loader,
        const std::vector<int32_t>& token_ids,
        float speaking_rate,
        ggml_backend_t backend) const;

    SanoPiperFrontOutput predict_latent(
        ModelLoader& loader,
        const std::vector<int32_t>& token_ids,
        const std::vector<int32_t>& durations,
        ggml_backend_t backend) const;

    std::vector<float> synthesize(
        ModelLoader& loader,
        const std::vector<int32_t>& token_ids,
        float speaking_rate,
        ggml_backend_t backend);

    const SanoPiperConfig& config() const { return config_; }

private:
    SanoPiperConfig config_;
    SanoPiperDecoder decoder_;
};

} // namespace inflect
