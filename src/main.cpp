#include "inflect-nano.h"
#include "utils.h"
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <cstring>
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
        }
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
    params.seed         = 1234;
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
