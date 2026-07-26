#pragma once

#include "model_config.h"
#include "model_loader.h"
#include "vocoder_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace inflect {

struct V2DurationFlowOutput {
    // Time-major [frames, channels], directly consumable by the decoder.
    std::vector<float> latent;
    std::vector<int32_t> durations;
    int frames = 0;
    int channels = 0;
};

class V2Model {
public:
    V2Model();
    ~V2Model() = default;

    bool load(ModelLoader& loader);
    bool load_duration(ModelLoader& loader);
    bool load_flow_block(ModelLoader& loader, int flow_index);
    bool load_decoder(ModelLoader& loader);
    V2DurationFlowOutput duration_and_flow(
        const std::vector<uint8_t>& blanked_tokens,
        float speed,
        float variation,
        uint64_t seed,
        const std::vector<float>* fixed_noise = nullptr,
        bool run_flow = true
    ) const;
    bool reverse_flow_block(V2DurationFlowOutput& output, int flow_index) const;
    int latent_channels() const { return latent_channels_; }

    void decode_streaming(
        const std::vector<float>& latent,
        int frames,
        int chunk_frames,
        ggml_backend_t backend,
        AudioCallback callback
    );

private:
    ModelLoader* loader_ = nullptr;
    std::unique_ptr<VocoderModel> decoder_;
    int sample_rate_ = 0;
    int latent_channels_ = 0;
    int hidden_channels_ = 0;
    int filter_channels_ = 0;
    int duration_hidden_channels_ = 0;
    int attention_heads_ = 0;
    int encoder_layers_ = 0;
    int upsample_initial_channels_ = 0;

    ggml_tensor* tensor(const std::string& name) const;
    bool configure_architecture(ModelLoader& loader);
    bool validate_tensor(const std::string& name) const;
};

} // namespace inflect
