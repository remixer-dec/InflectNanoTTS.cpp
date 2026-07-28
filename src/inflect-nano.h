#pragma once

#include "model_config.h"
#include "v2_frontend.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_buffer_type;

namespace inflect {

using BackendBufferType = ::ggml_backend_buffer_type*;

enum class ScratchMemoryKind {
    Default,
    Psram,
    InternalPreferred,
};

#if defined(INFLECT_LOW_MEMORY)
using QuantizeF32ToQ8Blocks32Fn = void (*)(
    const float* source,
    int8_t* destination,
    float* cached_scales,
    size_t blocks,
    bool skip_zero_blocks,
    uint64_t* max_cycles,
    uint64_t* scale_cycles,
    uint64_t* convert_cycles);
#endif

struct RuntimeConfig {
    BackendBufferType (*weight_buffer_type)() = nullptr;
    uint32_t (*now_ms)() = nullptr;
    uint32_t (*now_cycles)() = nullptr;
    void* (*scratch_alloc)(size_t bytes, ScratchMemoryKind kind) = nullptr;
    void (*scratch_free)(void* ptr) = nullptr;
    void (*trace_heap)(const char* label) = nullptr;
#if defined(INFLECT_LOW_MEMORY)
    void (*cooperate)() = nullptr;
    void (*dot_s8_blocks_32)(
        const int8_t* weights,
        const int8_t* inputs,
        int32_t* sums,
        size_t blocks,
        size_t rows) = nullptr;
    void (*dot_s8_scaled_blocks_32)(
        const int8_t* weights,
        const float* weight_scales,
        const int8_t* inputs,
        const float* input_scales,
        float* results,
        size_t blocks,
        size_t rows,
        uint64_t* dot_cycles,
        uint64_t* scale_cycles) = nullptr;
    void (*unpack_q4_0_blocks_32)(
        const uint8_t* packed_blocks,
        size_t packed_block_bytes,
        int8_t* values,
        uint16_t* scale_bits,
        size_t blocks) = nullptr;
    QuantizeF32ToQ8Blocks32Fn
        quantize_f32_to_q8_blocks_32 = nullptr;
    void (*store_zero_s8_blocks_32)(
        int8_t* destination,
        size_t blocks) = nullptr;
    int packed_quant_time_tile = 8;
#endif
    const char* backend_label = nullptr;
};

void configure_runtime(const RuntimeConfig& config);
const RuntimeConfig& runtime_config();
BackendBufferType runtime_weight_buffer_type();
uint32_t runtime_now_ms();
uint32_t runtime_now_cycles();
void* runtime_alloc_scratch(size_t bytes, ScratchMemoryKind kind);
void runtime_free_scratch(void* ptr);
void runtime_trace_heap(const char* label);
#if defined(INFLECT_LOW_MEMORY)
void runtime_cooperate();
void runtime_dot_s8_blocks_32(
    const int8_t* weights,
    const int8_t* inputs,
    int32_t* sums,
    size_t blocks,
    size_t rows);
void runtime_dot_s8_scaled_blocks_32(
    const int8_t* weights,
    const float* weight_scales,
    const int8_t* inputs,
    const float* input_scales,
    float* results,
    size_t blocks,
    size_t rows,
    uint64_t* dot_cycles,
    uint64_t* scale_cycles);
bool runtime_has_s8_dot_blocks_32();
bool runtime_has_s8_scaled_dot_blocks_32();
void runtime_unpack_q4_0_blocks_32(
    const uint8_t* packed_blocks,
    size_t packed_block_bytes,
    int8_t* values,
    uint16_t* scale_bits,
    size_t blocks);
void runtime_quantize_f32_to_q8_blocks_32(
    const float* source,
    int8_t* destination,
    float* cached_scales,
    size_t blocks,
    bool skip_zero_blocks,
    uint64_t* max_cycles,
    uint64_t* scale_cycles,
    uint64_t* convert_cycles);
void runtime_store_zero_s8_blocks_32(
    int8_t* destination,
    size_t blocks);
int runtime_packed_quant_time_tile();
#endif
const char* runtime_backend_label();

// ─────────────────────────────────────────────────────────────────────────
// Main synthesizer
// ─────────────────────────────────────────────────────────────────────────

class AcousticModel;
class VocoderModel;
class TextFrontend;
class ModelLoader;
class V2Model;

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
    bool load_v2(const std::string& model_gguf, const std::string& lexicon_bin);

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

    // Inflect v2 APIs. V1 state and behavior are independent.
    V2FrontendResult v2_text_to_tokens(const std::string& text);
    std::vector<float> synthesize_v2(
        const std::string& text,
        const V2SynthParams& params = {}
    );
    void synthesize_v2_streaming(
        const std::string& text,
        const V2SynthParams& params,
        AudioCallback callback
    );
    std::vector<float> synthesize_v2_unblanked_tokens(
        const std::vector<uint8_t>& tokens,
        const V2SynthParams& params = {}
    );
    std::vector<float> synthesize_v2_blanked_tokens(
        const std::vector<uint8_t>& tokens,
        const V2SynthParams& params = {}
    );
    void synthesize_v2_blanked_tokens_streaming(
        const std::vector<uint8_t>& tokens,
        const V2SynthParams& params,
        AudioCallback callback,
        const std::vector<float>* fixed_noise = nullptr
    );
    int v2_latent_channels() const;

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
    std::unique_ptr<ModelLoader> v2_loader_;
    std::unique_ptr<V2Frontend> v2_frontend_;
    std::unique_ptr<V2Model> v2_model_;
    std::string v2_model_path_;
};

} // namespace inflect
