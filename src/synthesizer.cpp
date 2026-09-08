#include "inflect-nano.h"
#include "acoustic_model.h"
#include "griffin_lim_vocoder.h"
#include "vocoder_model.h"
#include "text_frontend.h"
#include "v2_frontend.h"
#include "v2_model.h"
#include "v2_symbols.h"
#include "vocoder_quant_math.h"
#include "model_loader.h"
#include "sano_piper_model.h"
#include "memory_trace.h"
#include "utils.h"
#include "../ggml/include/ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>
#include <algorithm>

#ifndef INFLECT_VOCODER_BACKEND
#define INFLECT_VOCODER_BACKEND neural
#endif

#ifndef INFLECT_ACOUSTIC_SKIP_POSTNET
#define INFLECT_ACOUSTIC_SKIP_POSTNET 0
#endif

#define INFLECT_STRINGIFY_IMPL(x) #x
#define INFLECT_STRINGIFY(x) INFLECT_STRINGIFY_IMPL(x)

namespace inflect {

static ggml_backend_t g_backend = nullptr;
static int g_backend_threads = 1;
static RuntimeConfig g_runtime_config;

static uint32_t default_now_ms() {
    using clock = std::chrono::steady_clock;
    return (uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now().time_since_epoch()).count();
}

static uint32_t default_now_cycles() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock::now().time_since_epoch()).count());
}

void configure_runtime(const RuntimeConfig& config) {
    g_runtime_config = config;
}

const RuntimeConfig& runtime_config() {
    return g_runtime_config;
}

ggml_backend_buffer_type_t runtime_weight_buffer_type() {
    if (g_runtime_config.weight_buffer_type) {
        ggml_backend_buffer_type_t buft = g_runtime_config.weight_buffer_type();
        if (buft) {
            return buft;
        }
    }
    return ggml_backend_cpu_buffer_type();
}

uint32_t runtime_now_ms() {
    return g_runtime_config.now_ms ? g_runtime_config.now_ms() : default_now_ms();
}

uint32_t runtime_now_cycles() {
    return g_runtime_config.now_cycles
               ? g_runtime_config.now_cycles()
               : default_now_cycles();
}

void* runtime_alloc_scratch(size_t bytes, ScratchMemoryKind kind) {
    if (g_runtime_config.scratch_alloc) {
        return g_runtime_config.scratch_alloc(bytes, kind);
    }
    (void)kind;
    return std::malloc(bytes);
}

void runtime_free_scratch(void* ptr) {
    if (!ptr) {
        return;
    }
    if (g_runtime_config.scratch_free) {
        g_runtime_config.scratch_free(ptr);
        return;
    }
    std::free(ptr);
}

void runtime_trace_heap(const char* label) {
    if (g_runtime_config.trace_heap) {
        g_runtime_config.trace_heap(label);
    }
}

bool runtime_cancelled() {
    return g_runtime_config.cancelled &&
           g_runtime_config.cancelled(
               g_runtime_config.cancellation_user);
}

#if defined(INFLECT_LOW_MEMORY)
void runtime_cooperate() {
    if (g_runtime_config.cooperate) {
        g_runtime_config.cooperate();
    }
}

void runtime_dot_s8_blocks_32(
    const int8_t* weights,
    const int8_t* inputs,
    int32_t* sums,
    size_t blocks,
    size_t rows
) {
    if (g_runtime_config.dot_s8_blocks_32) {
        g_runtime_config.dot_s8_blocks_32(
            weights, inputs, sums, blocks, rows);
        return;
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks; ++block) {
            int32_t sum = 0;
            const int8_t* weight =
                weights + block * 32;
            const int8_t* input =
                inputs + (row * blocks + block) * 32;
            for (size_t i = 0; i < 32; ++i) {
                sum += static_cast<int32_t>(weight[i]) *
                       static_cast<int32_t>(input[i]);
            }
            sums[row * blocks + block] = sum;
        }
    }
}

void runtime_dot_s8_scaled_blocks_32(
    const int8_t* weights,
    const float* weight_scales,
    const int8_t* inputs,
    const float* input_scales,
    float* results,
    size_t blocks,
    size_t rows,
    bool skip_zero_scale_blocks,
    uint64_t* dot_cycles,
    uint64_t* scale_cycles
) {
    if (g_runtime_config.dot_s8_scaled_blocks_32) {
        g_runtime_config.dot_s8_scaled_blocks_32(
            weights,
            weight_scales,
            inputs,
            input_scales,
            results,
            blocks,
            rows,
            skip_zero_scale_blocks,
            dot_cycles,
            scale_cycles);
        return;
    }
    uint64_t measured_dot_cycles = 0;
    uint64_t measured_scale_cycles = 0;
    const bool profile =
        dot_cycles != nullptr || scale_cycles != nullptr;
    for (size_t row = 0; row < rows; ++row) {
        float result = 0.0f;
        for (size_t block = 0; block < blocks; ++block) {
            const size_t input_block = row * blocks + block;
            if (skip_zero_scale_blocks &&
                input_scales[input_block] == 0.0f) {
                continue;
            }
            const uint32_t dot_started =
                profile ? runtime_now_cycles() : 0;
            int32_t sum = 0;
            const int8_t* weight =
                weights + block * 32;
            const int8_t* input =
                inputs + input_block * 32;
            for (size_t i = 0; i < 32; ++i) {
                sum += static_cast<int32_t>(weight[i]) *
                       static_cast<int32_t>(input[i]);
            }
            const uint32_t dot_finished =
                profile ? runtime_now_cycles() : 0;
            result += static_cast<float>(sum) *
                      weight_scales[block] *
                      input_scales[input_block];
            if (profile) {
                const uint32_t scale_finished =
                    runtime_now_cycles();
                measured_dot_cycles +=
                    static_cast<uint32_t>(
                        dot_finished - dot_started);
                measured_scale_cycles +=
                    static_cast<uint32_t>(
                        scale_finished - dot_finished);
            }
        }
        results[row] = result;
    }
    if (dot_cycles) {
        *dot_cycles += measured_dot_cycles;
    }
    if (scale_cycles) {
        *scale_cycles += measured_scale_cycles;
    }
}

bool runtime_has_s8_dot_blocks_32() {
    return g_runtime_config.dot_s8_blocks_32 != nullptr;
}

bool runtime_has_s8_scaled_dot_blocks_32() {
    return g_runtime_config.dot_s8_scaled_blocks_32 != nullptr;
}

void runtime_unpack_q4_0_blocks_32(
    const uint8_t* packed_blocks,
    size_t packed_block_bytes,
    int8_t* values,
    uint16_t* scale_bits,
    size_t blocks
) {
    if (g_runtime_config.unpack_q4_0_blocks_32) {
        g_runtime_config.unpack_q4_0_blocks_32(
            packed_blocks,
            packed_block_bytes,
            values,
            scale_bits,
            blocks);
        return;
    }
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* source =
            packed_blocks + block * packed_block_bytes;
        std::memcpy(
            scale_bits + block, source, sizeof(uint16_t));
        source += sizeof(uint16_t);
        int8_t* destination = values + block * 32;
        for (size_t index = 0; index < 16; ++index) {
            destination[index] =
                static_cast<int8_t>(
                    (source[index] & 0x0fU) - 8);
            destination[index + 16] =
                static_cast<int8_t>(
                    (source[index] >> 4) - 8);
        }
    }
}

void runtime_quantize_f32_to_q8_blocks_32(
    const float* source,
    int8_t* destination,
    float* cached_scales,
    size_t blocks,
    bool skip_zero_blocks,
    uint64_t* max_cycles,
    uint64_t* scale_cycles,
    uint64_t* convert_cycles
) {
    if (g_runtime_config.quantize_f32_to_q8_blocks_32) {
        g_runtime_config.quantize_f32_to_q8_blocks_32(
            source,
            destination,
            cached_scales,
            blocks,
            skip_zero_blocks,
            max_cycles,
            scale_cycles,
            convert_cycles);
        return;
    }
    for (size_t block = 0; block < blocks; ++block) {
#if INFLECT_PROFILE_VOCODER_OPS && INFLECT_PROFILE_VOCODER_DETAIL
        const uint32_t max_started = runtime_now_cycles();
#endif
        const float maximum =
            vocoder_quant::absolute_max_32(source);
#if INFLECT_PROFILE_VOCODER_OPS && INFLECT_PROFILE_VOCODER_DETAIL
        const uint32_t max_finished = runtime_now_cycles();
#endif
        const float scale = maximum / 127.0f;
        const float inverse =
            scale != 0.0f ? 1.0f / scale : 0.0f;
        cached_scales[block] =
            vocoder_quant::cache_positive_scale_fp16(scale);
#if INFLECT_PROFILE_VOCODER_OPS && INFLECT_PROFILE_VOCODER_DETAIL
        const uint32_t scale_finished = runtime_now_cycles();
#endif
        if (skip_zero_blocks && maximum == 0.0f) {
            std::memset(destination, 0, 32);
        } else {
            for (int index = 0; index < 32; ++index) {
                destination[index] = static_cast<int8_t>(
                    vocoder_quant::
                        round_half_away_from_zero_bits_bounded(
                            source[index] * inverse));
            }
        }
#if INFLECT_PROFILE_VOCODER_OPS && INFLECT_PROFILE_VOCODER_DETAIL
        const uint32_t convert_finished = runtime_now_cycles();
        if (max_cycles != nullptr) {
            *max_cycles += static_cast<uint32_t>(
                max_finished - max_started);
        }
        if (scale_cycles != nullptr) {
            *scale_cycles += static_cast<uint32_t>(
                scale_finished - max_finished);
        }
        if (convert_cycles != nullptr) {
            *convert_cycles += static_cast<uint32_t>(
                convert_finished - scale_finished);
        }
#else
        (void)max_cycles;
        (void)scale_cycles;
        (void)convert_cycles;
#endif
        source += 32;
        destination += 32;
    }
}

void runtime_store_zero_s8_blocks_32(
    int8_t* destination,
    size_t blocks
) {
    if (g_runtime_config.store_zero_s8_blocks_32) {
        g_runtime_config.store_zero_s8_blocks_32(
            destination, blocks);
        return;
    }
    std::memset(destination, 0, blocks * 32);
}

int runtime_packed_quant_time_tile() {
    constexpr int kDefaultTimeTile = 8;
    constexpr int kMaxTimeTile = 64;
    const int configured = g_runtime_config.packed_quant_time_tile;
    if (configured <= 0) {
        return kDefaultTimeTile;
    }
    return std::min(configured, kMaxTimeTile);
}
#endif

const char* runtime_backend_label() {
    return (g_runtime_config.backend_label && g_runtime_config.backend_label[0])
               ? g_runtime_config.backend_label
               : "scalar";
}

static std::string debug_dump_dir() {
    const char* dir = std::getenv("INFLECT_DUMP_DIR");
    return (dir && dir[0]) ? std::string(dir) : std::string();
}

static std::string normalize_selector(std::string value) {
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1, value.size() - 2);
    }
    for (char& c : value) {
        c = (char)std::tolower((unsigned char)c);
        if (c == '-') c = '_';
    }
    return value;
}

static std::string build_vocoder_backend() {
    return normalize_selector(INFLECT_STRINGIFY(INFLECT_VOCODER_BACKEND));
}

static std::string selected_vocoder_backend(const SynthParams& params) {
    std::string backend = normalize_selector(params.vocoder_backend);
    if (backend.empty()) {
        backend = build_vocoder_backend();
    }
    if (backend != "neural" && backend != "griffin_lim") {
        fprintf(stderr,
                "[Synthesizer] Unsupported vocoder backend=%s; falling back to neural\n",
                backend.c_str());
        backend = "neural";
    }
    return backend;
}

static void ensure_debug_dir(const std::string& dir) {
    if (dir.empty()) return;
    std::string cur;
    for (char c : dir) {
        cur.push_back(c);
        if (c == '/') {
            if (cur.size() > 1) mkdir(cur.c_str(), 0755);
        }
    }
    mkdir(dir.c_str(), 0755);
}

static std::string debug_path(const std::string& dir, const std::string& name) {
    return dir + "/" + name;
}

static void debug_manifest(const std::string& dir, const std::string& line, bool reset = false) {
    if (dir.empty()) return;
    std::ofstream f(debug_path(dir, "manifest.txt"), reset ? std::ios::trunc : std::ios::app);
    f << line << "\n";
}

static void debug_save_f32(const std::string& dir, const std::string& name,
                           const std::vector<float>& data, const std::string& shape) {
    if (dir.empty()) return;
    save_bin(debug_path(dir, name + ".f32"), data.data(), data.size());
    debug_manifest(dir, name + " f32 " + shape + " count=" + std::to_string(data.size()));
}

// Convert 2D tensor from GGML layout (ne0 fastest) to numpy/C layout matching
// Python golden dumps, which store [A, B] in C-order.
// Input: data in GGML layout a + b*A. Output: [A, B] C-order.
static std::vector<float> debug_transpose_2d(const std::vector<float>& data, int A, int B) {
    std::vector<float> out(A * B);
    for (int a = 0; a < A; a++) {
        for (int b = 0; b < B; b++) {
            out[a * B + b] = data[a + b * A];
        }
    }
    return out;
}

static void debug_save_i32(const std::string& dir, const std::string& name,
                           const std::vector<int32_t>& data, const std::string& shape) {
    if (dir.empty()) return;
    std::ofstream f(debug_path(dir, name + ".i32"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int32_t));
    debug_manifest(dir, name + " i32 " + shape + " count=" + std::to_string(data.size()));
}

static bool backend_abort_callback(void*) {
#if defined(INFLECT_LOW_MEMORY)
    runtime_cooperate();
#endif
    return runtime_cancelled();
}

void Synthesizer::init_backend(int n_threads) {
    if (g_backend) return;
    g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!g_backend) {
        fprintf(stderr, "[Synthesizer] Failed to init backend\n");
        return;
    }
    int threads = n_threads;
    if (threads <= 0) {
        threads = (int)std::thread::hardware_concurrency();
        if (threads <= 0) {
            threads = 1;
        }
    }
    if (ggml_backend_is_cpu(g_backend)) {
        g_backend_threads = threads;
        ggml_backend_cpu_set_n_threads(g_backend, threads);
        ggml_backend_cpu_set_abort_callback(
            g_backend, backend_abort_callback, nullptr);
    }
    fprintf(stderr, "[Synthesizer] Backend initialized threads=%d simd=%s\n",
            threads,
            runtime_backend_label());
    mem_trace_rss("after backend init");
}

void Synthesizer::set_backend_threads(int n_threads) {
    if (!g_backend || !ggml_backend_is_cpu(g_backend) || n_threads <= 0) {
        return;
    }
    ggml_backend_cpu_set_n_threads(g_backend, n_threads);
    g_backend_threads = n_threads;
    fprintf(stderr, "[Synthesizer] Backend threads set=%d\n", n_threads);
}

void Synthesizer::free_backend() {
    if (g_backend) {
        ggml_backend_free(g_backend);
        g_backend = nullptr;
    }
}

Synthesizer::Synthesizer() {
    init_backend();
}

Synthesizer::~Synthesizer() = default;

bool Synthesizer::load_acoustic(const std::string& path) {
    acoustic_path_ = path;
    acoustic_loader_ = std::make_unique<ModelLoader>();
#if defined(INFLECT_LOW_MEMORY)
    static const std::vector<std::string> encoder_prefixes = {
        "phone.", "tone.", "lang.", "speaker.weight", "speaker_proj.",
        "encoder.", "duration_head.", "energy_head.", "bright_head.",
        "pitch_head.", "energy_proj.", "bright_proj.", "pitch_proj.",
        "abs_frame.", "frame_proj.", "local_ctx.",
    };
    if (!acoustic_loader_->load_selected(path, encoder_prefixes)) return false;
#else
    if (!acoustic_loader_->load(path)) return false;
#endif

    AcousticConfig cfg;
    cfg.vocab_size     = acoustic_loader_->get_i32("vocab_size", 256);
    cfg.tone_size      = acoustic_loader_->get_i32("tone_size", 16);
    cfg.lang_size      = acoustic_loader_->get_i32("lang_size", 4);
    cfg.n_mels         = acoustic_loader_->get_i32("n_mels", 80);
    cfg.hidden         = acoustic_loader_->get_i32("hidden", 168);
    cfg.encoder_layers = acoustic_loader_->get_i32("encoder_layers", 5);
    cfg.decoder_layers = acoustic_loader_->get_i32("decoder_layers", 6);
    cfg.encoder_ff_mult = acoustic_loader_->get_i32("encoder_ff_mult", 4);
    cfg.decoder_ff_mult = acoustic_loader_->get_i32("decoder_ff_mult", 3);
    cfg.kernel_size    = acoustic_loader_->get_i32("kernel_size", 7);
    cfg.speaker_count  = acoustic_loader_->get_i32("speaker_count", 2);
    cfg.speaker_dim    = acoustic_loader_->get_i32("speaker_dim", 64);
    cfg.abs_frame_bins = acoustic_loader_->get_i32("abs_frame_bins", 512);
    cfg.sample_rate    = acoustic_loader_->get_i32("sample_rate", 24000);
    cfg.max_frames     = acoustic_loader_->get_i32("max_frames", 1400);
    cfg.postnet_scale  = acoustic_loader_->get_f32("postnet_scale", 0.1f);
    acoustic_config_ = cfg;
    sample_rate_ = cfg.sample_rate;

    acoustic_ = std::make_unique<AcousticModel>(cfg);
#if defined(INFLECT_LOW_MEMORY)
    const bool ok = acoustic_->load_encoder(*acoustic_loader_);
#else
    const bool ok = acoustic_->load(*acoustic_loader_);
#endif
    if (!ok) {
        acoustic_.reset();
        acoustic_loader_.reset();
    }
    mem_trace_rss("after acoustic load");
    return ok;
}

bool Synthesizer::load_vocoder(const std::string& path) {
#if defined(INFLECT_LOW_MEMORY)
    deferred_vocoder_path_ = path;
    vocoder_.reset();
    vocoder_loader_.reset();
    fprintf(stderr, "[Synthesizer] Deferred vocoder load for low-memory mode\n");
    mem_trace_rss("after vocoder defer");
    return true;
#else
    return load_vocoder_now(path);
#endif
}

bool Synthesizer::load_vocoder_now(const std::string& path) {
    vocoder_loader_ = std::make_unique<ModelLoader>();
    if (!vocoder_loader_->load(path)) return false;

    VocoderConfig cfg;
    cfg.sample_rate = vocoder_loader_->get_i32("sample_rate", 24000);
    cfg.num_mels    = vocoder_loader_->get_i32("num_mels", 80);
    cfg.upsample_initial_channel = vocoder_loader_->get_i32("upsample_initial_channel", 144);
    cfg.activation  = vocoder_loader_->get_string("activation", "snake");

    vocoder_ = std::make_unique<VocoderModel>(cfg);
    const bool ok = vocoder_->load(*vocoder_loader_);
    mem_trace_rss("after vocoder load");
    return ok;
}

bool Synthesizer::load_cmudict(const std::string& path) {
    frontend_ = std::make_unique<TextFrontend>();
    const bool ok = frontend_->load_cmudict(path);
    mem_trace_rss("after cmudict load");
    return ok;
}

bool Synthesizer::load_v2(const std::string& model_path,
                          const std::string& lexicon_path) {
    auto loader = std::make_unique<ModelLoader>();
#if defined(INFLECT_LOW_MEMORY)
    static const std::vector<std::string> duration_prefixes = {
        "enc_p.", "dp.",
    };
    if (!loader->load_selected(model_path, duration_prefixes)) return false;
#else
    if (!loader->load(model_path)) return false;
#endif
    auto frontend = std::make_unique<V2Frontend>();
    if (!frontend->load_lexicon(lexicon_path)) return false;
    if (loader->get_string("inflect.v2.symbol_hash") != v2::kSymbolHashHex) {
        fprintf(stderr,
                "[Synthesizer] V2 model/lexicon symbol hash mismatch model=%s runtime=%s\n",
                loader->get_string("inflect.v2.symbol_hash", "<missing>").c_str(),
                v2::kSymbolHashHex);
        return false;
    }
    auto model = std::make_unique<V2Model>();
#if defined(INFLECT_LOW_MEMORY)
    if (!model->load_duration(*loader)) return false;
#else
    if (!model->load(*loader)) return false;
#endif
    v2_loader_ = std::move(loader);
    v2_frontend_ = std::move(frontend);
    v2_model_ = std::move(model);
    v2_model_path_ = model_path;
    sample_rate_ = 24000;
    fprintf(stderr,
            "[Synthesizer] Loaded Inflect v2 model=%s lexicon=%s symbols=%s"
            " weight_mode=staged\n",
            model_path.c_str(), lexicon_path.c_str(), v2::kSymbolHashHex);
    mem_trace_rss("after v2 load");
    return true;
}

bool Synthesizer::load_sano(const std::string& model_path,
                            const std::string& lexicon_path) {
    auto loader = std::make_unique<ModelLoader>();
    if (!loader->open(model_path)) return false;

    auto model = std::make_unique<SanoPiperModel>();
    if (!model->configure(*loader)) {
        fprintf(stderr, "[Synthesizer] Sano model metadata is invalid\n");
        return false;
    }

    auto frontend = std::make_unique<SanoFrontend>();
    if (!frontend->load_lexicon(lexicon_path, model->config())) {
        fprintf(stderr, "[Synthesizer] Failed to load Sano lexicon\n");
        return false;
    }

    sample_rate_ = model->config().sample_rate;
    sano_model_path_ = model_path;
    sano_loader_ = std::move(loader);
    sano_frontend_ = std::move(frontend);
    sano_model_ = std::move(model);
    fprintf(stderr,
            "[Synthesizer] Loaded Sano Piperlite model=%s lexicon=%s "
            "voice=%s language=%s sample_rate=%d\n",
            model_path.c_str(), lexicon_path.c_str(),
            sano_model_->config().voice.c_str(),
            sano_model_->config().language.c_str(), sample_rate_);
    mem_trace_rss("after sano load");
    return true;
}

V2FrontendResult Synthesizer::v2_text_to_tokens(const std::string& text) {
    if (!v2_frontend_) {
        fprintf(stderr, "[Synthesizer] V2 frontend not loaded\n");
        return {};
    }
    return v2_frontend_->process(text);
}

int Synthesizer::v2_latent_channels() const {
    return v2_model_ ? v2_model_->latent_channels() : 0;
}

SanoFrontendResult Synthesizer::sano_text_to_tokens(const std::string& text) {
    if (!sano_frontend_ || !sano_model_) {
        fprintf(stderr, "[Synthesizer] Sano frontend/model not loaded\n");
        return {};
    }
    return sano_frontend_->process(text, sano_model_->config());
}

std::vector<float> Synthesizer::synthesize_sano(
    const std::string& text,
    const SanoSynthParams& params
) {
    if (!g_backend || !sano_frontend_ || !sano_model_ || !sano_loader_) {
        fprintf(stderr, "[Synthesizer] Sano model/backend not loaded\n");
        return {};
    }
    if (params.speaking_rate < 0.5f || params.speaking_rate > 2.0f ||
        !std::isfinite(params.speaking_rate)) {
        fprintf(stderr,
                "[Synthesizer] Sano speaking_rate out of range: %.3f\n",
                params.speaking_rate);
        return {};
    }

    const uint32_t frontend_started = runtime_now_ms();
    const SanoFrontendResult frontend = sano_text_to_tokens(text);
    fprintf(stderr, "[SanoStage] frontend tokens=%zu elapsed_ms=%u\n",
            frontend.tokens.size(),
            static_cast<unsigned>(runtime_now_ms() - frontend_started));
    for (const auto& word : frontend.approximated_words) {
        fprintf(stderr, "[SanoFrontend] OOV pronunciation fallback: %s\n", word.c_str());
    }
    for (const auto& word : frontend.spelled_words) {
        fprintf(stderr, "[SanoFrontend] OOV spelling fallback: %s\n", word.c_str());
    }
    for (const auto& word : frontend.unpronounceable_words) {
        fprintf(stderr,
                "[SanoFrontend] cannot spell OOV word '%s'; rebuild the lexicon "
                "with alphabet coverage or add the word with --text/--supplemental\n",
                word.c_str());
    }
    if (!frontend.ok || frontend.tokens.empty()) {
        fprintf(stderr, "[Synthesizer] Sano frontend failed; audio not generated\n");
        return {};
    }

    const uint32_t started_ms = runtime_now_ms();
    SanoThreadpoolScope threadpool(g_backend, g_backend_threads);
    std::vector<float> audio = sano_model_->synthesize(
        *sano_loader_, frontend.tokens, params.speaking_rate, g_backend);
    fprintf(stderr,
            "[SanoStage] tokens=%zu samples=%zu speaking_rate=%.3f elapsed_ms=%u\n",
            frontend.tokens.size(), audio.size(), params.speaking_rate,
            static_cast<unsigned>(runtime_now_ms() - started_ms));
    return audio;
}

static std::vector<std::string> split_v2_text(const std::string& input,
                                               size_t limit = 280) {
    std::string normalized;
    normalized.reserve(input.size());
    bool space = false;
    for (unsigned char c : input) {
        if (std::isspace(c)) {
            space = !normalized.empty();
        } else {
            if (space) normalized.push_back(' ');
            normalized.push_back(static_cast<char>(c));
            space = false;
        }
    }
    std::vector<std::string> sentences;
    size_t start = 0;
    for (size_t i = 0; i < normalized.size(); ++i) {
        if (std::strchr(".!?;:", normalized[i]) &&
            i + 1 < normalized.size() && normalized[i + 1] == ' ') {
            sentences.push_back(normalized.substr(start, i + 1 - start));
            start = i + 2;
            i = start ? start - 1 : 0;
        }
    }
    if (start < normalized.size()) sentences.push_back(normalized.substr(start));
    if (sentences.empty() && !normalized.empty()) sentences.push_back(normalized);

    std::vector<std::string> chunks;
    for (std::string sentence : sentences) {
        while (sentence.size() > limit) {
            const size_t search_end = std::min(sentence.size(), limit + 1);
            size_t punctuation = std::string::npos;
            for (char mark : {',', ';', ':'}) {
                const size_t found = sentence.rfind(mark, search_end - 1);
                if (found != std::string::npos &&
                    (punctuation == std::string::npos || found > punctuation)) {
                    punctuation = found;
                }
            }
            size_t split = punctuation != std::string::npos &&
                                   punctuation >= limit / 2
                               ? punctuation + 1
                               : sentence.rfind(' ', search_end - 1);
            if (split == std::string::npos || split < limit / 2) split = limit;
            chunks.push_back(sentence.substr(0, split));
            sentence.erase(0, split);
            while (!sentence.empty() && sentence.front() == ' ') sentence.erase(0, 1);
        }
        if (!sentence.empty()) chunks.push_back(std::move(sentence));
    }
    return chunks;
}

static float v2_boundary_pause(const std::string& chunk) {
    const char ending = chunk.empty() ? '\0' : chunk.back();
    switch (ending) {
        case '?': return 0.28f;
        case '!': return 0.24f;
        case '.': return 0.22f;
        case ';': return 0.16f;
        case ':': return 0.13f;
        case ',': return 0.09f;
        default: return 0.08f;
    }
}

struct V2FadeSink {
    AudioCallback callback;
    std::vector<float> pending;
    size_t start_index = 0;
    static constexpr size_t kFadeSamples = 120;

    explicit V2FadeSink(AudioCallback sink) : callback(std::move(sink)) {}

    void write(const float* samples, size_t count) {
        pending.reserve(pending.size() + count);
        for (size_t i = 0; i < count; ++i, ++start_index) {
            float value = std::max(-1.0f, std::min(1.0f, samples[i]));
            if (start_index < kFadeSamples) {
                const float gain = static_cast<float>(start_index) /
                                   static_cast<float>(kFadeSamples - 1);
                value *= gain;
            }
            pending.push_back(value);
        }
        if (pending.size() > kFadeSamples) {
            const size_t emit = pending.size() - kFadeSamples;
            callback(pending.data(), emit);
            pending.erase(pending.begin(), pending.begin() + emit);
        }
    }

    void finish() {
        const size_t fade = std::min(kFadeSamples, pending.size() / 2);
        if (fade > 0) {
            for (size_t i = 0; i < fade; ++i) {
                const float gain = static_cast<float>(fade - 1 - i) /
                                   static_cast<float>(std::max<size_t>(1, fade - 1));
                pending[pending.size() - fade + i] *= gain;
            }
        }
        if (!pending.empty()) callback(pending.data(), pending.size());
        pending.clear();
    }
};

void Synthesizer::synthesize_v2_blanked_tokens_streaming(
    const std::vector<uint8_t>& tokens,
    const V2SynthParams& params,
    AudioCallback callback,
    const std::vector<float>* fixed_noise
) {
    if (!g_backend || !callback
#if defined(INFLECT_LOW_MEMORY)
        || v2_model_path_.empty()
#else
        || !v2_model_
#endif
    ) {
        fprintf(stderr, "[Synthesizer] V2 model/backend not loaded\n");
        return;
    }
    if (params.speed < 0.5f || params.speed > 2.0f ||
        params.variation < 0.0f || params.variation > 1.0f) {
        fprintf(stderr,
                "[Synthesizer] V2 controls out of range speed=%.3f variation=%.3f\n",
                params.speed, params.variation);
        return;
    }
    if (tokens.empty()) {
        fprintf(stderr, "[Synthesizer] Empty V2 token sequence\n");
        return;
    }
    for (uint8_t token : tokens) {
        if (token >= v2::kSymbolCount) {
            fprintf(stderr, "[Synthesizer] V2 token ID out of range: %u\n", token);
            return;
        }
    }
#if defined(INFLECT_LOW_MEMORY)
    if (!v2_model_ || !v2_loader_) {
        static const std::vector<std::string> duration_prefixes = {
            "enc_p.", "dp.",
        };
        v2_loader_ = std::make_unique<ModelLoader>();
        v2_model_ = std::make_unique<V2Model>();
        if (!v2_loader_->load_selected(
                v2_model_path_, duration_prefixes) ||
            !v2_model_->load_duration(*v2_loader_)) {
            fprintf(stderr, "[Synthesizer] Failed to load V2 duration stage\n");
            v2_model_.reset();
            v2_loader_.reset();
            return;
        }
    }
#endif
    const uint32_t duration_started_ms = runtime_now_ms();
    V2DurationFlowOutput latent = v2_model_->duration_and_flow(
        tokens, params.speed, params.variation, params.seed, fixed_noise,
#if defined(INFLECT_LOW_MEMORY)
        false
#else
        true
#endif
    );
    fprintf(stderr,
            "[V2Stage] duration compute_ms=%u frames=%d channels=%d\n",
            static_cast<unsigned>(
                runtime_now_ms() - duration_started_ms),
            latent.frames,
            latent.channels);
    if (runtime_cancelled()) {
#if defined(INFLECT_LOW_MEMORY)
        v2_model_.reset();
        v2_loader_.reset();
        mem_release_to_os();
#endif
        fprintf(stderr, "[Synthesizer] V2 duration cancelled\n");
        return;
    }
    if (latent.frames <= 0) {
#if defined(INFLECT_LOW_MEMORY)
        v2_model_.reset();
        v2_loader_.reset();
#endif
        return;
    }
#if defined(INFLECT_LOW_MEMORY)
    const bool use_staged_decoder =
        latent.channels >= 192;
    const bool group_flow_weights = !use_staged_decoder;
    runtime_trace_heap("v2 duration complete");

    uint32_t grouped_flow_load_ms = 0;
    if (group_flow_weights) {
        const uint32_t load_started_ms = runtime_now_ms();
        if (!v2_loader_->select({"flow.flows."})) {
            fprintf(stderr, "[Synthesizer] Failed grouped V2 flow load\n");
            v2_model_.reset();
            v2_loader_.reset();
            return;
        }
        grouped_flow_load_ms = runtime_now_ms() - load_started_ms;
        fprintf(
            stderr,
            "[V2Stage] grouped flow load_ms=%u\n",
            static_cast<unsigned>(grouped_flow_load_ms));
        runtime_trace_heap("v2 grouped flows loaded");
    }
    for (int flow = 6; flow >= 0; flow -= 2) {
        if (runtime_cancelled()) {
            v2_model_.reset();
            v2_loader_.reset();
            mem_release_to_os();
            fprintf(stderr, "[Synthesizer] V2 flow cancelled\n");
            return;
        }
        const std::string prefix =
            "flow.flows." + std::to_string(flow) + ".";
        const uint32_t load_started_ms = runtime_now_ms();
        if ((!group_flow_weights && !v2_loader_->select({prefix})) ||
            !v2_model_->load_flow_block(*v2_loader_, flow)) {
            fprintf(stderr, "[Synthesizer] Failed V2 flow load %d\n", flow);
            v2_model_.reset();
            v2_loader_.reset();
            return;
        }
        const uint32_t compute_started_ms = runtime_now_ms();
        if (!v2_model_->reverse_flow_block(latent, flow)) {
            fprintf(stderr, "[Synthesizer] Failed V2 reverse flow block %d\n", flow);
            v2_model_.reset();
            v2_loader_.reset();
            return;
        }
        if (runtime_cancelled()) {
            v2_model_.reset();
            v2_loader_.reset();
            mem_release_to_os();
            fprintf(stderr, "[Synthesizer] V2 flow cancelled\n");
            return;
        }
        fprintf(
            stderr,
            "[V2Stage] flow=%d load_ms=%u compute_ms=%u\n",
            flow,
            static_cast<unsigned>(
                group_flow_weights
                    ? 0
                    : compute_started_ms - load_started_ms),
            static_cast<unsigned>(
                runtime_now_ms() - compute_started_ms));
        if (!group_flow_weights) {
            runtime_trace_heap(
                ("v2 flow " + std::to_string(flow) + " complete").c_str());
        }
    }
    if (use_staged_decoder) {
        if (!v2_model_->prepare_staged_decoder()) {
            fprintf(
                stderr,
                "[Synthesizer] Failed to prepare staged V2 decoder\n");
            v2_model_.reset();
            return;
        }
        v2_loader_.reset();
        mem_release_to_os();
        runtime_trace_heap("v2 acoustic weights released");
    } else {
        const uint32_t decoder_load_started_ms = runtime_now_ms();
        if (!v2_loader_->select({"dec."}) ||
            !v2_model_->load_decoder(*v2_loader_)) {
            fprintf(
                stderr,
                "[Synthesizer] Failed to load V2 decoder stage\n");
            v2_model_.reset();
            v2_loader_.reset();
            return;
        }
        fprintf(
            stderr,
            "[V2Stage] decoder load_ms=%u\n",
            static_cast<unsigned>(
                runtime_now_ms() - decoder_load_started_ms));
    }
#endif
    V2FadeSink sink{std::move(callback)};
#if defined(INFLECT_LOW_MEMORY)
    if (use_staged_decoder) {
        if (!v2_model_->decode_staged(
                v2_model_path_,
                latent.latent,
                latent.frames,
                g_backend,
                [&](const float* samples, size_t count) {
                    sink.write(samples, count);
                })) {
            fprintf(
                stderr,
                "[Synthesizer] Staged V2 decoder failed\n");
        }
    } else {
        v2_model_->decode_streaming(
            latent.latent,
            latent.frames,
            params.decoder_chunk_frames,
            g_backend,
            [&](const float* samples, size_t count) {
                sink.write(samples, count);
            });
    }
#else
    v2_model_->decode_streaming(
        latent.latent, latent.frames, params.decoder_chunk_frames, g_backend,
        [&](const float* samples, size_t count) { sink.write(samples, count); });
#endif
    sink.finish();
#if defined(INFLECT_LOW_MEMORY)
    v2_model_.reset();
    v2_loader_.reset();
    mem_release_to_os();
    runtime_trace_heap("v2 decoder released");
#endif
}

std::vector<float> Synthesizer::synthesize_v2_blanked_tokens(
    const std::vector<uint8_t>& tokens,
    const V2SynthParams& params
) {
    std::vector<float> audio;
    synthesize_v2_blanked_tokens_streaming(
        tokens, params,
        [&](const float* samples, size_t count) {
            audio.insert(audio.end(), samples, samples + count);
        });
    return audio;
}

std::vector<float> Synthesizer::synthesize_v2_unblanked_tokens(
    const std::vector<uint8_t>& tokens,
    const V2SynthParams& params
) {
    return synthesize_v2_blanked_tokens(v2::intersperse_blanks(tokens), params);
}

void Synthesizer::synthesize_v2_streaming(
    const std::string& text,
    const V2SynthParams& params,
    AudioCallback callback
) {
    if (!v2_frontend_ || !callback) {
        fprintf(stderr, "[Synthesizer] V2 frontend not loaded\n");
        return;
    }
    const auto chunks = split_v2_text(text);
    if (chunks.empty()) {
        fprintf(stderr, "[Synthesizer] V2 text must not be empty\n");
        return;
    }
    std::vector<float> silence;
    for (size_t index = 0; index < chunks.size(); ++index) {
        if (index > 0) {
            const size_t count = static_cast<size_t>(
                std::lround(24000.0f * v2_boundary_pause(chunks[index - 1])));
            silence.assign(count, 0.0f);
            callback(silence.data(), silence.size());
        }
        V2FrontendResult frontend = v2_frontend_->process(chunks[index]);
        if (frontend.blanked_tokens.empty()) continue;
        V2SynthParams chunk_params = params;
        chunk_params.seed += index;
        synthesize_v2_blanked_tokens_streaming(
            frontend.blanked_tokens, chunk_params, callback);
    }
}

std::vector<float> Synthesizer::synthesize_v2(
    const std::string& text,
    const V2SynthParams& params
) {
    std::vector<float> audio;
    synthesize_v2_streaming(
        text, params,
        [&](const float* samples, size_t count) {
            audio.insert(audio.end(), samples, samples + count);
        });
    return audio;
}

TokenSequence Synthesizer::text_to_tokens(const std::string& text) {
    if (!frontend_) {
        fprintf(stderr, "[Synthesizer] Text frontend not loaded\n");
        return {};
    }
    auto result = frontend_->process(text);
    return {result.phone_ids, result.tone_ids, result.lang_ids};
}

std::vector<float> Synthesizer::synthesize(
    const std::string& text,
    const SynthParams& params
) {
    std::vector<float> audio;
    synthesize_streaming(text, params, [&](const float* samples, size_t n) {
        audio.insert(audio.end(), samples, samples + n);
    }, params.vocoder_chunk_frames);

    const std::string dump_dir = debug_dump_dir();
    if (!dump_dir.empty()) {
        debug_save_f32(dump_dir, "audio_raw", audio, "[" + std::to_string(audio.size()) + "]");
    }

    if (selected_vocoder_backend(params) == "griffin_lim") {
        fprintf(stderr, "[Synthesizer] Global audio normalization skipped backend=griffin_lim\n");
    } else {
        normalize_audio(audio);
    }

    if (!dump_dir.empty()) {
        debug_save_f32(dump_dir, "audio_normalized", audio, "[" + std::to_string(audio.size()) + "]");
    }

    return audio;
}

void Synthesizer::synthesize_streaming(
    const std::string& text,
    const SynthParams& params,
    AudioCallback callback,
    int vocoder_chunk_frames
) {
    const uint32_t synth_start_ms = runtime_now_ms();
    uint32_t stage_start_ms = synth_start_ms;
    const std::string vocoder_backend = selected_vocoder_backend(params);
    fprintf(stderr, "[Synthesizer] Vocoder backend=%s\n", vocoder_backend.c_str());
    if (!frontend_ || !acoustic_) {
        fprintf(stderr, "[Synthesizer] V1 frontend/acoustic model not loaded\n");
        return;
    }
    const std::string dump_dir = debug_dump_dir();
    if (!dump_dir.empty()) {
        ensure_debug_dir(dump_dir);
        debug_manifest(dump_dir, "inflect-nano debug dump", true);
        debug_manifest(dump_dir, "text " + text);
    }

    // ── 1. Text → Tokens ────────────────────────────────────────────
    auto tokens = text_to_tokens(text);
    if (tokens.phone_ids.empty()) {
        fprintf(stderr, "[Synthesizer] No tokens generated\n");
        return;
    }

    fprintf(stderr, "[Synthesizer] %zu tokens text_ms=%u total_ms=%u\n",
            tokens.phone_ids.size(),
            (unsigned)(runtime_now_ms() - stage_start_ms),
            (unsigned)(runtime_now_ms() - synth_start_ms));
    stage_start_ms = runtime_now_ms();
    if (!dump_dir.empty()) {
        const std::string shape = "[" + std::to_string(tokens.phone_ids.size()) + "]";
        debug_save_i32(dump_dir, "phone_ids", tokens.phone_ids, shape);
        debug_save_i32(dump_dir, "tone_ids", tokens.tone_ids, shape);
        debug_save_i32(dump_dir, "lang_ids", tokens.lang_ids, shape);
    }

    // ── 2. Graph 1: Encoder + Prediction Heads ──────────────────────
    auto enc_out = acoustic_->run_encoder(
        tokens.phone_ids, tokens.tone_ids, tokens.lang_ids,
        params.speaker_id, g_backend
    );
    if (runtime_cancelled() || enc_out.encoded.empty()) {
        fprintf(stderr, "[Synthesizer] Encoder stopped\n");
        return;
    }
#if defined(INFLECT_LOW_MEMORY)
    mem_release_to_os();
#endif
    mem_trace_rss("after encoder");
    mem_trace_heap("after encoder");

    fprintf(stderr, "[Synthesizer] Encoder done: %d frames predicted stage_ms=%u total_ms=%u\n",
            enc_out.seq_len,
            (unsigned)(runtime_now_ms() - stage_start_ms),
            (unsigned)(runtime_now_ms() - synth_start_ms));
    stage_start_ms = runtime_now_ms();
    if (!dump_dir.empty()) {
        debug_save_f32(dump_dir, "encoded",
                       debug_transpose_2d(enc_out.encoded, enc_out.hidden, enc_out.seq_len),
                       "[" + std::to_string(enc_out.hidden) + "," + std::to_string(enc_out.seq_len) + "]");
        debug_save_f32(dump_dir, "embed_sum",
                       debug_transpose_2d(enc_out.embed_sum, enc_out.hidden, enc_out.seq_len),
                       "[" + std::to_string(enc_out.hidden) + "," + std::to_string(enc_out.seq_len) + "]");
        for (size_t i = 0; i < enc_out.enc_blocks.size(); i++) {
            debug_save_f32(dump_dir, "enc_block_" + std::to_string(i),
                           debug_transpose_2d(enc_out.enc_blocks[i], enc_out.hidden, enc_out.seq_len),
                           "[" + std::to_string(enc_out.hidden) + "," + std::to_string(enc_out.seq_len) + "]");
        }
        debug_save_f32(dump_dir, "log_durations", enc_out.log_durations,
                       "[" + std::to_string(enc_out.seq_len) + "]");
        debug_save_f32(dump_dir, "energy", enc_out.energy,
                       "[" + std::to_string(enc_out.seq_len) + "]");
        debug_save_f32(dump_dir, "bright", enc_out.bright,
                       "[" + std::to_string(enc_out.seq_len) + "]");
        debug_save_f32(dump_dir, "pitch",
                       debug_transpose_2d(enc_out.pitch, 2, enc_out.seq_len),
                       "[2," + std::to_string(enc_out.seq_len) + "]");
    }

    // ── 3. CPU Bridge: Length Regulation ────────────────────────────
    auto features = acoustic_->length_regulate(
        enc_out, params.length_scale, params.pitch_scale, params.energy_scale
    );
#if defined(INFLECT_LOW_MEMORY)
    mem_release_to_os();
#endif
    mem_trace_rss("after length regulation");
    mem_trace_heap("after length regulation");

    fprintf(stderr, "[Synthesizer] Length regulated: %d frames stage_ms=%u total_ms=%u\n",
            features.n_frames,
            (unsigned)(runtime_now_ms() - stage_start_ms),
            (unsigned)(runtime_now_ms() - synth_start_ms));
    stage_start_ms = runtime_now_ms();
    if (!dump_dir.empty()) {
        debug_save_i32(dump_dir, "durations", features.durations,
                       "[" + std::to_string(features.durations.size()) + "]");
        debug_save_f32(dump_dir, "regulated",
                       debug_transpose_2d(features.features, features.hidden, features.n_frames),
                       "[" + std::to_string(features.hidden) + "," + std::to_string(features.n_frames) + "]");
    }

#if defined(INFLECT_LOW_MEMORY)
    enc_out = EncoderOutput{};
    acoustic_.reset();
    acoustic_loader_->release_selected();
    mem_release_to_os();
    mem_trace_rss("after acoustic encoder release");

    std::vector<std::string> decoder_prefixes = {
        "decoder.", "frame_gru.", "mel_head.",
    };
#if !INFLECT_ACOUSTIC_SKIP_POSTNET
    decoder_prefixes.push_back("postnet.");
#endif
    if (!acoustic_loader_->select(decoder_prefixes)) {
        fprintf(
            stderr,
            "[Synthesizer] Failed to reload decoder-only acoustic model\n");
        return;
    }
    acoustic_ = std::make_unique<AcousticModel>(acoustic_config_);
    if (!acoustic_->load_decoder(*acoustic_loader_)) {
        fprintf(
            stderr,
            "[Synthesizer] Failed to bind decoder-only acoustic model\n");
        return;
    }
    mem_trace_rss("after acoustic decoder reload");
    mem_trace_heap("after acoustic decoder reload");
    fprintf(
        stderr,
        "[Synthesizer] Acoustic decoder reloaded stage_ms=%u total_ms=%u\n",
        (unsigned)(runtime_now_ms() - stage_start_ms),
        (unsigned)(runtime_now_ms() - synth_start_ms));
    stage_start_ms = runtime_now_ms();
#endif

    // ── 4. Graph 2: Decoder → Mel ───────────────────────────────────
    auto mel = acoustic_->run_decoder(features, g_backend);
    if (runtime_cancelled() || mel.empty()) {
        fprintf(stderr, "[Synthesizer] Decoder stopped\n");
        return;
    }
    mem_trace_rss("after decoder");
    mem_trace_heap("after decoder");
    int n_mels = acoustic_->config().n_mels;
    int n_frames = features.n_frames;

    fprintf(stderr, "[Synthesizer] Mel generated: %zu values stage_ms=%u total_ms=%u\n",
            mel.size(),
            (unsigned)(runtime_now_ms() - stage_start_ms),
            (unsigned)(runtime_now_ms() - synth_start_ms));
    stage_start_ms = runtime_now_ms();
    if (!dump_dir.empty()) {
        debug_save_f32(dump_dir, "mel",
                       debug_transpose_2d(mel, n_mels, n_frames),
                       "[" + std::to_string(n_mels) + "," + std::to_string(n_frames) + "]");
    }

#if defined(INFLECT_LOW_MEMORY)
    features = RegulatedFeatures{};
    acoustic_.reset();
    acoustic_loader_.reset();
    mem_release_to_os();
    mem_trace_rss("after acoustic release");
    mem_trace_heap("after acoustic release");
    if (vocoder_backend == "griffin_lim") {
        fprintf(stderr, "[Synthesizer] Vocoder model load skipped backend=griffin_lim\n");
    } else if (!vocoder_ && !deferred_vocoder_path_.empty()) {
        if (!load_vocoder_now(deferred_vocoder_path_)) {
            fprintf(stderr, "[Synthesizer] Failed to load deferred vocoder\n");
            return;
        }
    }
    mem_trace_heap("after vocoder ready");
    fprintf(stderr, "[Synthesizer] Vocoder ready stage_ms=%u total_ms=%u\n",
            (unsigned)(runtime_now_ms() - stage_start_ms),
            (unsigned)(runtime_now_ms() - synth_start_ms));
    stage_start_ms = runtime_now_ms();
#endif

    // ── 5. Graph 3: Vocoder → Audio ─────────────────────────────────
    if (vocoder_backend == "griffin_lim") {
        GriffinLimConfig gl_cfg;
        gl_cfg.sample_rate = acoustic_config_.sample_rate;
        gl_cfg.n_mels = n_mels;
        gl_cfg.iterations = params.griffin_lim_iterations;
        gl_cfg.seed = params.seed;
        griffin_lim_vocode_streaming(mel, n_mels, n_frames, gl_cfg, callback);
    } else {
        if (!vocoder_) {
            fprintf(stderr, "[Synthesizer] Vocoder not loaded\n");
            return;
        }
        vocoder_->vocode_streaming(
            mel, n_mels, n_frames, vocoder_chunk_frames, g_backend, callback
        );
    }
    mem_trace_rss("after vocoder");
    mem_trace_heap("after vocoder");
    fprintf(stderr, "[Synthesizer] Vocoder done stage_ms=%u total_ms=%u\n",
            (unsigned)(runtime_now_ms() - stage_start_ms),
            (unsigned)(runtime_now_ms() - synth_start_ms));
}

} // namespace inflect
