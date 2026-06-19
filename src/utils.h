#pragma once

#include <cstdint>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

namespace inflect {

// ─────────────────────────────────────────────────────────────────────────
// Deterministic RNG (Philox-style for reproducibility)
// ─────────────────────────────────────────────────────────────────────────

class DeterministicRNG {
public:
    explicit DeterministicRNG(uint64_t seed = 1234) : state_(seed) {}

    void seed(uint64_t s) { state_ = s; }

    // Uniform [0, 1)
    float uniform() {
        // Simple xorshift64* → uniform float
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        uint64_t result = state_ * 0x2545F4914F6CDD1DULL;
        return static_cast<float>(result >> 40) * (1.0f / (1ULL << 24));
    }

    // Standard normal (Box-Muller)
    float randn() {
        float u1 = uniform();
        float u2 = uniform();
        // Avoid log(0)
        u1 = std::max(u1, 1e-7f);
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979f * u2);
    }

private:
    uint64_t state_;
};

// ─────────────────────────────────────────────────────────────────────────
// WAV file writer
// ─────────────────────────────────────────────────────────────────────────

#pragma pack(push, 1)
struct WAVHeader {
    char     riff[4]     = {'R','I','F','F'};
    uint32_t wav_size;
    char     wave[4]     = {'W','A','V','E'};
    char     fmt[4]      = {'f','m','t',' '};
    uint32_t fmt_size    = 16;
    uint16_t audio_fmt   = 1;  // PCM
    uint16_t channels    = 1;
    uint32_t sample_rate = 24000;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bit_depth   = 16;
    char     data[4]     = {'d','a','t','a'};
    uint32_t data_size;
};
#pragma pack(pop)

inline bool write_wav(
    const std::string& path,
    const std::vector<float>& samples,
    int sample_rate = 24000
) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    WAVHeader hdr;
    hdr.sample_rate = sample_rate;
    hdr.channels    = 1;
    hdr.bit_depth   = 16;
    hdr.block_align = hdr.channels * hdr.bit_depth / 8;
    hdr.byte_rate   = hdr.sample_rate * hdr.block_align;
    hdr.data_size   = samples.size() * hdr.block_align;
    hdr.wav_size    = 4 + (8 + 16) + (8 + hdr.data_size);

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        float s = std::max(-1.0f, std::min(1.0f, samples[i]));
        pcm[i] = static_cast<int16_t>(s * 32767.0f);
    }
    f.write(reinterpret_cast<const char*>(pcm.data()), pcm.size() * sizeof(int16_t));

    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Audio normalization (port from inference.py)
// ─────────────────────────────────────────────────────────────────────────

inline float rms_db(const std::vector<float>& audio) {
    double sum_sq = 0.0;
    for (float s : audio) sum_sq += (double)s * s;
    float rms = std::sqrt(sum_sq / std::max((size_t)1, audio.size()));
    return 20.0f * std::log10(rms + 1e-9f);
}

inline void normalize_audio(
    std::vector<float>& audio,
    float target_rms_db = -20.0f,
    float peak_db = -1.0f
) {
    if (audio.empty()) return;

    // Remove DC offset
    float mean = 0.0f;
    for (float s : audio) mean += s;
    mean /= audio.size();
    for (float& s : audio) s -= mean;

    // RMS normalization
    float current_rms = rms_db(audio);
    float gain = std::pow(10.0f, (target_rms_db - current_rms) / 20.0f);
    for (float& s : audio) s *= gain;

    // Peak limiting
    float peak = 0.0f;
    for (float s : audio) peak = std::max(peak, std::abs(s));
    float peak_limit = std::pow(10.0f, peak_db / 20.0f);
    if (peak > peak_limit) {
        float scale = peak_limit / (peak + 1e-9f);
        for (float& s : audio) s *= scale;
    }

    // Clamp
    for (float& s : audio) s = std::max(-1.0f, std::min(1.0f, s));
}

// ─────────────────────────────────────────────────────────────────────────
// Golden reference testing
// ─────────────────────────────────────────────────────────────────────────

inline std::vector<float> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t size = f.tellg() / sizeof(float);
    f.seekg(0);
    std::vector<float> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    return data;
}

inline bool save_bin(const std::string& path, const float* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    return true;
}

inline bool tensor_close(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float max_abs_error = 1e-4f,
    float min_cosine = 0.999f
) {
    if (a.size() != b.size()) return false;

    float max_err = 0.0f;
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        float err = std::abs(a[i] - b[i]);
        max_err = std::max(max_err, err);
        dot    += (double)a[i] * b[i];
        norm_a += (double)a[i] * a[i];
        norm_b += (double)b[i] * b[i];
    }

    float cosine = (float)(dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10));
    return max_err < max_abs_error && cosine > min_cosine;
}

} // namespace inflect
