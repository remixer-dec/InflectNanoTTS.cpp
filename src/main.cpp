#include "inflect-nano.h"
#include "utils.h"
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#ifndef INFLECT_VOCODER_BACKEND
#define INFLECT_VOCODER_BACKEND neural
#endif

#ifndef INFLECT_GRIFFIN_LIM_ITERS
#define INFLECT_GRIFFIN_LIM_ITERS 8
#endif

#define INFLECT_MAIN_STRINGIFY_IMPL(x) #x
#define INFLECT_MAIN_STRINGIFY(x) INFLECT_MAIN_STRINGIFY_IMPL(x)

static std::string normalize_backend(std::string value) {
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1, value.size() - 2);
    }
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') {
            c = '_';
        }
    }
    if (value == "gl" || value == "griffinlim") {
        return "griffin_lim";
    }
    return value;
}

static bool is_griffin_lim_backend(const std::string& backend) {
    return backend == "griffin_lim";
}

int main(int argc, char** argv) {
    using namespace inflect;

    std::string text = "Hello, this is a test of the Inflect Nano text to speech system.";
    std::string acoustic_path = "inflect_acoustic.gguf";
    std::string vocoder_path  = "inflect_vocoder.gguf";
    std::string cmudict_path  = "cmudict.bin";
    std::string output_path   = "output.wav";
    std::string model_family = "v1";
    std::string v2_model_path;
    std::string v2_lexicon_path;
    std::string sano_model_path = "sano_piperlite.gguf";
    std::string sano_lexicon_path = "sano_lexicon.snl";
    std::string token_file;
    std::string noise_file;
    bool tokens_blanked = false;
    float v2_speed = 1.0f;
    float v2_variation = 0.667f;
    float sano_speaking_rate = 1.0f;
    uint64_t seed = 1234;
    int decoder_chunk_frames = 32;
    int n_threads = -1;
    int vocoder_chunk_frames = 0;
    int griffin_lim_iterations = INFLECT_GRIFFIN_LIM_ITERS;
    std::string vocoder_backend =
        normalize_backend(INFLECT_MAIN_STRINGIFY(INFLECT_VOCODER_BACKEND));

    if (const char* env = std::getenv("INFLECT_THREADS")) {
        int parsed = std::atoi(env);
        if (parsed > 0) n_threads = parsed;
    }
    if (const char* env = std::getenv("INFLECT_VOCODER_BACKEND")) {
        vocoder_backend = normalize_backend(env);
    }
    if (const char* env = std::getenv("INFLECT_VOCODER_CHUNK_FRAMES")) {
        int parsed = std::atoi(env);
        if (parsed > 0) vocoder_chunk_frames = parsed;
    }
    if (const char* env = std::getenv("INFLECT_GRIFFIN_LIM_ITERS")) {
        int parsed = std::atoi(env);
        if (parsed >= 0) griffin_lim_iterations = parsed;
    }
#if defined(INFLECT_LOW_MEMORY)
    if (vocoder_chunk_frames <= 0) {
        vocoder_chunk_frames = 11;
    }
#endif

    // Parse args (simplified)
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-t" && i + 1 < argc) {
            text = argv[++i];
        } else if (arg == "-a" && i + 1 < argc) {
            acoustic_path = argv[++i];
        } else if (arg == "-v" && i + 1 < argc) {
            vocoder_path = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {
            cmudict_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((arg == "-j" || arg == "--threads") && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (arg == "--vocoder-chunk-frames" && i + 1 < argc) {
            vocoder_chunk_frames = std::atoi(argv[++i]);
        } else if (arg == "--vocoder-backend" && i + 1 < argc) {
            vocoder_backend = normalize_backend(argv[++i]);
        } else if (arg == "--griffin-lim-iters" && i + 1 < argc) {
            griffin_lim_iterations = std::atoi(argv[++i]);
        } else if (arg == "--model-family" && i + 1 < argc) {
            model_family = argv[++i];
        } else if (arg == "--v2-model" && i + 1 < argc) {
            v2_model_path = argv[++i];
            model_family = "v2";
        } else if (arg == "--v2-lexicon" && i + 1 < argc) {
            v2_lexicon_path = argv[++i];
        } else if (arg == "--sano-model" && i + 1 < argc) {
            sano_model_path = argv[++i];
            model_family = "sano";
        } else if (arg == "--sano-lexicon" && i + 1 < argc) {
            sano_lexicon_path = argv[++i];
        } else if (arg == "--speaking-rate" && i + 1 < argc) {
            sano_speaking_rate = std::strtof(argv[++i], nullptr);
        } else if (arg == "--speed" && i + 1 < argc) {
            v2_speed = std::strtof(argv[++i], nullptr);
        } else if (arg == "--variation" && i + 1 < argc) {
            v2_variation = std::strtof(argv[++i], nullptr);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--decoder-chunk-frames" && i + 1 < argc) {
            decoder_chunk_frames = std::atoi(argv[++i]);
        } else if (arg == "--token-file" && i + 1 < argc) {
            token_file = argv[++i];
        } else if (arg == "--noise-file" && i + 1 < argc) {
            noise_file = argv[++i];
        } else if (arg == "--tokens-blanked") {
            tokens_blanked = true;
        }
    }
    if (model_family == "piperlite") model_family = "sano";
    if (model_family != "v1" && model_family != "v2" &&
        model_family != "sano") {
        fprintf(stderr,
                "Unsupported model family '%s'; expected v1, v2, or sano\n",
                model_family.c_str());
        return 1;
    }
    if (vocoder_backend != "neural" && vocoder_backend != "griffin_lim") {
        fprintf(stderr,
                "Unsupported vocoder backend '%s'; expected neural or griffin_lim\n",
                vocoder_backend.c_str());
        return 1;
    }

    // Init
    Synthesizer::init_backend(n_threads);

    Synthesizer synth;
    if (model_family == "v2") {
        if (v2_model_path.empty() || v2_lexicon_path.empty()) {
            fprintf(stderr,
                    "V2 requires --v2-model MODEL.gguf and "
                    "--v2-lexicon lexicon.bin\n");
            return 1;
        }
        if (!synth.load_v2(v2_model_path, v2_lexicon_path)) {
            fprintf(stderr, "Failed to load Inflect v2 assets\n");
            return 1;
        }
        V2SynthParams params;
        params.speed = v2_speed;
        params.variation = v2_variation;
        params.seed = seed;
        params.decoder_chunk_frames = decoder_chunk_frames;
        std::vector<float> audio;
        if (!token_file.empty()) {
            std::ifstream source(token_file);
            if (!source) {
                fprintf(stderr, "Failed to open token file: %s\n", token_file.c_str());
                return 1;
            }
            std::vector<uint8_t> tokens;
            std::string field;
            while (source >> field) {
                size_t start = 0;
                while (start < field.size()) {
                    const size_t comma = field.find(',', start);
                    const std::string item =
                        field.substr(start, comma == std::string::npos
                                               ? std::string::npos
                                               : comma - start);
                    if (!item.empty()) {
                        char* end = nullptr;
                        const long value = std::strtol(item.c_str(), &end, 10);
                        if (!end || *end != '\0' || value < 0 || value >= 178) {
                            fprintf(stderr, "Invalid V2 token ID: %s\n", item.c_str());
                            return 1;
                        }
                        tokens.push_back(static_cast<uint8_t>(value));
                    }
                    if (comma == std::string::npos) break;
                    start = comma + 1;
                }
            }
            if (!noise_file.empty()) {
                if (!tokens_blanked) {
                    fprintf(stderr,
                            "--noise-file requires --tokens-blanked for golden parity\n");
                    return 1;
                }
                std::ifstream noise_source(
                    noise_file, std::ios::binary | std::ios::ate);
                if (!noise_source) {
                    fprintf(stderr, "Failed to open noise file: %s\n",
                            noise_file.c_str());
                    return 1;
                }
                const std::streamsize bytes = noise_source.tellg();
                const size_t channels =
                    static_cast<size_t>(synth.v2_latent_channels());
                if (bytes <= 0 ||
                    channels == 0 ||
                    bytes % static_cast<std::streamsize>(
                                channels * sizeof(float)) != 0) {
                    fprintf(stderr,
                            "Invalid golden noise size: %lld bytes\n",
                            static_cast<long long>(bytes));
                    return 1;
                }
                noise_source.seekg(0);
                std::vector<float> channel_major(
                    static_cast<size_t>(bytes) / sizeof(float));
                noise_source.read(
                    reinterpret_cast<char*>(channel_major.data()), bytes);
                if (!noise_source) {
                    fprintf(stderr, "Failed to read noise file: %s\n",
                            noise_file.c_str());
                    return 1;
                }
                const size_t frames = channel_major.size() / channels;
                std::vector<float> time_major(channel_major.size());
                for (size_t channel = 0; channel < channels; ++channel) {
                    for (size_t frame = 0; frame < frames; ++frame) {
                        time_major[frame * channels + channel] =
                            channel_major[channel * frames + frame];
                    }
                }
                synth.synthesize_v2_blanked_tokens_streaming(
                    tokens, params,
                    [&](const float* samples, size_t count) {
                        audio.insert(audio.end(), samples, samples + count);
                    },
                    &time_major);
            } else {
                audio = tokens_blanked
                            ? synth.synthesize_v2_blanked_tokens(tokens, params)
                            : synth.synthesize_v2_unblanked_tokens(tokens, params);
            }
        } else {
            audio = synth.synthesize_v2(text, params);
        }
        fprintf(stderr,
                "Generated V2 %zu samples (%.2f seconds) speed=%.3f "
                "variation=%.3f seed=%llu\n",
                audio.size(), static_cast<float>(audio.size()) / synth.sample_rate(),
                params.speed, params.variation,
                static_cast<unsigned long long>(params.seed));
        if (!write_wav(output_path, audio, synth.sample_rate())) {
            fprintf(stderr, "Failed to write %s\n", output_path.c_str());
            return 1;
        }
        fprintf(stderr, "Wrote %s\n", output_path.c_str());
        Synthesizer::free_backend();
        return 0;
    }

    if (model_family == "sano") {
        if (!synth.load_sano(sano_model_path, sano_lexicon_path)) {
            fprintf(stderr, "Failed to load Sano Piperlite assets\n");
            Synthesizer::free_backend();
            return 1;
        }
        SanoSynthParams params;
        params.speaking_rate = sano_speaking_rate;
        fprintf(stderr, "Synthesizing Sano Piperlite: %s\n", text.c_str());
        const std::vector<float> audio = synth.synthesize_sano(text, params);
        if (audio.empty()) {
            fprintf(stderr, "Sano synthesis failed; output file not written\n");
            Synthesizer::free_backend();
            return 1;
        }
        fprintf(stderr,
                "Generated Sano %zu samples (%.2f seconds) rate=%d\n",
                audio.size(),
                static_cast<float>(audio.size()) / synth.sample_rate(),
                synth.sample_rate());
        if (!write_wav(output_path, audio, synth.sample_rate())) {
            fprintf(stderr, "Failed to write %s\n", output_path.c_str());
            Synthesizer::free_backend();
            return 1;
        }
        fprintf(stderr, "Wrote %s\n", output_path.c_str());
        Synthesizer::free_backend();
        return 0;
    }

    if (!synth.load_acoustic(acoustic_path)) {
        fprintf(stderr, "Failed to load acoustic model\n");
        return 1;
    }
    if (!is_griffin_lim_backend(vocoder_backend) && !synth.load_vocoder(vocoder_path)) {
        fprintf(stderr, "Failed to load vocoder\n");
        return 1;
    } else if (is_griffin_lim_backend(vocoder_backend)) {
        fprintf(stderr, "Skipping neural vocoder load; backend=griffin_lim\n");
    }
    if (!synth.load_cmudict(cmudict_path)) {
        fprintf(stderr, "Warning: cmudict not loaded, text frontend unavailable\n");
    }

    // Synthesize
    SynthParams params;
    params.length_scale = 1.0f;
    params.pitch_scale  = 1.0f;
    params.energy_scale = 1.0f;
    params.speaker_id   = 0;
    params.seed         = seed;
    params.vocoder_chunk_frames = vocoder_chunk_frames;
    params.vocoder_backend = vocoder_backend;
    params.griffin_lim_iterations = griffin_lim_iterations;

    fprintf(stderr, "Synthesizing backend=%s: %s\n",
            vocoder_backend.c_str(), text.c_str());
    auto audio = synth.synthesize(text, params);

    fprintf(stderr, "Generated %zu samples (%.2f seconds)\n",
            audio.size(), (float)audio.size() / synth.sample_rate());

    // Write WAV
    if (write_wav(output_path, audio, synth.sample_rate())) {
        fprintf(stderr, "Wrote %s\n", output_path.c_str());
    }

    Synthesizer::free_backend();
    return 0;
}
