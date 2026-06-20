#include "inflect-nano.h"
#include "acoustic_model.h"
#include "vocoder_model.h"
#include "text_frontend.h"
#include "model_loader.h"
#include "memory_trace.h"
#include "utils.h"
#include <ggml-cpu.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>

namespace inflect {

static ggml_backend_t g_backend = nullptr;

static std::string debug_dump_dir() {
    const char* dir = std::getenv("INFLECT_DUMP_DIR");
    return (dir && dir[0]) ? std::string(dir) : std::string();
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

void Synthesizer::init_backend(int n_threads) {
    if (g_backend) return;
    g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!g_backend) {
        fprintf(stderr, "[Synthesizer] Failed to init backend\n");
        return;
    }
    if (ggml_backend_is_cpu(g_backend)) {
        ggml_backend_cpu_set_n_threads(g_backend,
            n_threads > 0 ? n_threads : (int)std::thread::hardware_concurrency());
    }
    fprintf(stderr, "[Synthesizer] Backend initialized\n");
    mem_trace_rss("after backend init");
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
    acoustic_loader_ = std::make_unique<ModelLoader>();
    if (!acoustic_loader_->load(path)) return false;

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

    acoustic_ = std::make_unique<AcousticModel>(cfg);
    const bool ok = acoustic_->load(*acoustic_loader_);
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

    normalize_audio(audio);

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

    fprintf(stderr, "[Synthesizer] %zu tokens\n", tokens.phone_ids.size());
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
    mem_trace_rss("after encoder");

    fprintf(stderr, "[Synthesizer] Encoder done: %d frames predicted\n",
            enc_out.seq_len);
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
    mem_trace_rss("after length regulation");

    fprintf(stderr, "[Synthesizer] Length regulated: %d frames\n",
            features.n_frames);
    if (!dump_dir.empty()) {
        debug_save_i32(dump_dir, "durations", features.durations,
                       "[" + std::to_string(features.durations.size()) + "]");
        debug_save_f32(dump_dir, "regulated",
                       debug_transpose_2d(features.features, features.hidden, features.n_frames),
                       "[" + std::to_string(features.hidden) + "," + std::to_string(features.n_frames) + "]");
    }

    // ── 4. Graph 2: Decoder → Mel ───────────────────────────────────
    auto mel = acoustic_->run_decoder(features, g_backend);
    mem_trace_rss("after decoder");
    int n_mels = acoustic_->config().n_mels;
    int n_frames = features.n_frames;

#if defined(INFLECT_LOW_MEMORY)
    enc_out = EncoderOutput{};
    features = RegulatedFeatures{};
    acoustic_.reset();
    acoustic_loader_.reset();
    mem_trace_rss("after acoustic release");
    if (!vocoder_ && !deferred_vocoder_path_.empty()) {
        if (!load_vocoder_now(deferred_vocoder_path_)) {
            fprintf(stderr, "[Synthesizer] Failed to load deferred vocoder\n");
            return;
        }
    }
#endif

    fprintf(stderr, "[Synthesizer] Mel generated: %zu values\n", mel.size());
    if (!dump_dir.empty()) {
        debug_save_f32(dump_dir, "mel",
                       debug_transpose_2d(mel, n_mels, n_frames),
                       "[" + std::to_string(n_mels) + "," + std::to_string(n_frames) + "]");
    }

    // ── 5. Graph 3: Vocoder → Audio ─────────────────────────────────
    vocoder_->vocode_streaming(
        mel, n_mels, n_frames, vocoder_chunk_frames, g_backend, callback
    );
    mem_trace_rss("after vocoder");
}

} // namespace inflect
