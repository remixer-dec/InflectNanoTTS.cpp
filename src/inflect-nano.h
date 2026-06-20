#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include "model_config.h"
#include <memory>
#include <string>
#include <vector>

namespace inflect {

// ─────────────────────────────────────────────────────────────────────────
// Main synthesizer
// ─────────────────────────────────────────────────────────────────────────

class AcousticModel;
class VocoderModel;
class TextFrontend;
class ModelLoader;

class Synthesizer {
public:
    Synthesizer();
    ~Synthesizer();

    Synthesizer(const Synthesizer&) = delete;
    Synthesizer& operator=(const Synthesizer&) = delete;

    // Load model files
    bool load_acoustic(const std::string& gguf_path);
    bool load_vocoder(const std::string& gguf_path);
    bool load_cmudict(const std::string& bin_path);

    // Full synthesis: text → audio
    std::vector<float> synthesize(
        const std::string& text,
        const SynthParams& params = {}
    );

    // Streaming: calls callback for each vocoder chunk
    void synthesize_streaming(
        const std::string& text,
        const SynthParams& params,
        AudioCallback callback,
        int vocoder_chunk_frames = 0   // 0 = no chunking
    );

    // Text → tokens (exposed for debugging / golden reference tests)
    TokenSequence text_to_tokens(const std::string& text);

    int sample_rate() const { return 24000; }

    // Backend management
    static void init_backend(int n_threads = -1);
    static void free_backend();

private:
    std::unique_ptr<AcousticModel> acoustic_;
    std::unique_ptr<VocoderModel>  vocoder_;
    std::unique_ptr<TextFrontend>  frontend_;
    std::unique_ptr<ModelLoader>   acoustic_loader_;
    std::unique_ptr<ModelLoader>   vocoder_loader_;
    std::string acoustic_path_;
    AcousticConfig acoustic_config_;
    std::string deferred_vocoder_path_;
    bool load_vocoder_now(const std::string& path);
};

} // namespace inflect
