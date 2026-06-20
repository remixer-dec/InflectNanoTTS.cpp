#include "inflect-nano.h"
#include "utils.h"
#include <cstdlib>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    using namespace inflect;

    std::string text = "Hello, this is a test of the Inflect Nano text to speech system.";
    std::string acoustic_path = "inflect_acoustic.gguf";
    std::string vocoder_path  = "inflect_vocoder.gguf";
    std::string cmudict_path  = "cmudict.bin";
    std::string output_path   = "output.wav";
    int n_threads = -1;
    int vocoder_chunk_frames = 0;

    if (const char* env = std::getenv("INFLECT_THREADS")) {
        int parsed = std::atoi(env);
        if (parsed > 0) n_threads = parsed;
    }
    if (const char* env = std::getenv("INFLECT_VOCODER_CHUNK_FRAMES")) {
        int parsed = std::atoi(env);
        if (parsed > 0) vocoder_chunk_frames = parsed;
    }
#if defined(INFLECT_LOW_MEMORY)
    if (vocoder_chunk_frames <= 0) {
        vocoder_chunk_frames = 11;
    }
#endif

    // Parse args (simplified)
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-t" && i + 1 < argc) text = argv[++i];
        if (std::string(argv[i]) == "-a" && i + 1 < argc) acoustic_path = argv[++i];
        if (std::string(argv[i]) == "-v" && i + 1 < argc) vocoder_path = argv[++i];
        if (std::string(argv[i]) == "-d" && i + 1 < argc) cmudict_path = argv[++i];
        if (std::string(argv[i]) == "-o" && i + 1 < argc) output_path = argv[++i];
        if ((std::string(argv[i]) == "-j" || std::string(argv[i]) == "--threads") && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--vocoder-chunk-frames" && i + 1 < argc) {
            vocoder_chunk_frames = std::atoi(argv[++i]);
        }
    }

    // Init
    Synthesizer::init_backend(n_threads);

    Synthesizer synth;
    if (!synth.load_acoustic(acoustic_path)) {
        fprintf(stderr, "Failed to load acoustic model\n");
        return 1;
    }
    if (!synth.load_vocoder(vocoder_path)) {
        fprintf(stderr, "Failed to load vocoder\n");
        return 1;
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

    fprintf(stderr, "Synthesizing: %s\n", text.c_str());
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
