#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include "model_config.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace inflect {

enum class ScratchMemoryKind {
    Default,
    Psram,
    InternalPreferred,
};

struct RuntimeConfig {
    ggml_backend_buffer_type_t (*weight_buffer_type)() = nullptr;
    uint32_t (*now_ms)() = nullptr;
    void* (*scratch_alloc)(size_t bytes, ScratchMemoryKind kind) = nullptr;
    void (*scratch_free)(void* ptr) = nullptr;
    void (*trace_heap)(const char* label) = nullptr;
    const char* backend_label = nullptr;
};

void configure_runtime(const RuntimeConfig& config);
const RuntimeConfig& runtime_config();
ggml_backend_buffer_type_t runtime_weight_buffer_type();
uint32_t runtime_now_ms();
void* runtime_alloc_scratch(size_t bytes, ScratchMemoryKind kind);
void runtime_free_scratch(void* ptr);
void runtime_trace_heap(const char* label);
const char* runtime_backend_label();

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
    static void set_backend_threads(int n_threads);
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
