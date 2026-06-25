#include "griffin_lim_vocoder.h"
#include "inflect-nano.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#if __has_include(<esp_dsp.h>)
#include <esp_dsp.h>
#define INFLECT_GL_HAS_ESP_DSP 1
#else
#define INFLECT_GL_HAS_ESP_DSP 0
#endif

namespace inflect {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr int kFftGuardFloats = 16;

#ifndef INFLECT_GRIFFIN_LIM_USE_ESP_DSP
#define INFLECT_GRIFFIN_LIM_USE_ESP_DSP 0
#endif

struct FloatStats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float rms = 0.0f;
    float peak = 0.0f;
};

static uint32_t now_ms() { return runtime_now_ms(); }

static float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

static float mel_value_to_magnitude(float v) {
    // Acoustic mel values are log-like for the neural vocoder. Clamp before
    // exp so a bad frame cannot explode the Griffin-Lim overlap-add buffer.
    v = std::max(-12.0f, std::min(4.0f, v));
    return std::exp(v);
}

static FloatStats compute_stats(const std::vector<float>& values) {
    FloatStats s;
    if (values.empty()) {
        return s;
    }
    s.min = values[0];
    s.max = values[0];
    double sum = 0.0;
    double sum_sq = 0.0;
    float peak = 0.0f;
    for (float v : values) {
        if (!std::isfinite(v)) {
            continue;
        }
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        sum += v;
        sum_sq += (double)v * (double)v;
        peak = std::max(peak, std::fabs(v));
    }
    s.mean = (float)(sum / (double)values.size());
    s.rms = (float)std::sqrt(sum_sq / (double)values.size());
    s.peak = peak;
    return s;
}

static void log_stats(const char* label, const std::vector<float>& values) {
    const FloatStats s = compute_stats(values);
    fprintf(stderr,
            "[GriffinLim] %s min=%.6g max=%.6g mean=%.6g rms=%.6g peak=%.6g\n",
            label,
            s.min,
            s.max,
            s.mean,
            s.rms,
            s.peak);
}

static bool fft_radix2_inplace(std::vector<float>& data, int n_fft) {
    if (n_fft <= 1 || (n_fft & (n_fft - 1)) != 0 || (int)data.size() < 2 * n_fft) {
        return false;
    }

    for (int i = 1, j = 0; i < n_fft; i++) {
        int bit = n_fft >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[2 * i], data[2 * j]);
            std::swap(data[2 * i + 1], data[2 * j + 1]);
        }
    }

    for (int len = 2; len <= n_fft; len <<= 1) {
        const float angle = -2.0f * kPi / (float)len;
        const float wlen_r = std::cos(angle);
        const float wlen_i = std::sin(angle);
        for (int i = 0; i < n_fft; i += len) {
            float wr = 1.0f;
            float wi = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                const int u = i + j;
                const int v = i + j + len / 2;
                const float ur = data[2 * u];
                const float ui = data[2 * u + 1];
                const float vr = data[2 * v] * wr - data[2 * v + 1] * wi;
                const float vi = data[2 * v] * wi + data[2 * v + 1] * wr;
                data[2 * u] = ur + vr;
                data[2 * u + 1] = ui + vi;
                data[2 * v] = ur - vr;
                data[2 * v + 1] = ui - vi;

                const float next_wr = wr * wlen_r - wi * wlen_i;
                wi = wr * wlen_i + wi * wlen_r;
                wr = next_wr;
            }
        }
    }
    return true;
}

static bool fft_inplace(std::vector<float>& data, int n_fft, bool inverse, bool use_esp_dsp) {
    const int active_floats = 2 * n_fft;
    if ((int)data.size() < active_floats) {
        return false;
    }
    if ((int)data.size() > active_floats) {
        std::fill(data.begin() + active_floats, data.end(), 0.0f);
    }

    if (inverse) {
        for (int i = 0; i < n_fft; i++) {
            data[2 * i + 1] = -data[2 * i + 1];
        }
    }

#if INFLECT_GL_HAS_ESP_DSP && INFLECT_GRIFFIN_LIM_USE_ESP_DSP
    if (use_esp_dsp) {
        if (dsps_fft2r_fc32(data.data(), n_fft) == ESP_OK) {
            dsps_bit_rev_fc32(data.data(), n_fft);
        } else {
            use_esp_dsp = false;
        }
    }
#endif

    if (!use_esp_dsp) {
        if (!fft_radix2_inplace(data, n_fft)) {
            return false;
        }
    }

    if (inverse) {
        const float scale = 1.0f / (float)n_fft;
        for (int i = 0; i < n_fft; i++) {
            data[2 * i] *= scale;
            data[2 * i + 1] = -data[2 * i + 1] * scale;
        }
    }
    return true;
}

static std::vector<float> make_hann_window(int n_fft, int win_length) {
    std::vector<float> window(n_fft, 0.0f);
    if (win_length <= 1) {
        return window;
    }
    const int offset = std::max(0, (n_fft - win_length) / 2);
    for (int i = 0; i < win_length && offset + i < n_fft; i++) {
        window[offset + i] = 0.5f - 0.5f * std::cos(2.0f * kPi * (float)i / (float)(win_length - 1));
    }
    return window;
}

static void make_mel_filterbank(
    int n_mels,
    int n_bins,
    int sample_rate,
    int n_fft,
    float f_min,
    float f_max,
    std::vector<float>& filterbank,
    std::vector<float>& bin_weight_sum
) {
    filterbank.assign(n_mels * n_bins, 0.0f);
    bin_weight_sum.assign(n_bins, 0.0f);

    f_max = std::min(f_max, 0.5f * (float)sample_rate);
    const float mel_min = hz_to_mel(std::max(0.0f, f_min));
    const float mel_max = hz_to_mel(std::max(f_min + 1.0f, f_max));

    std::vector<int> edges(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        const float mel = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mels + 1);
        const float hz = mel_to_hz(mel);
        edges[i] = std::max(0, std::min(n_bins - 1, (int)std::floor(((float)n_fft + 1.0f) * hz / sample_rate)));
    }

    for (int m = 0; m < n_mels; m++) {
        const int left = edges[m];
        const int center = std::max(edges[m + 1], left + 1);
        const int right = std::max(edges[m + 2], center + 1);

        for (int k = left; k < center && k < n_bins; k++) {
            const float w = (float)(k - left) / (float)(center - left);
            filterbank[m * n_bins + k] = w;
            bin_weight_sum[k] += w;
        }
        for (int k = center; k < right && k < n_bins; k++) {
            const float w = (float)(right - k) / (float)(right - center);
            filterbank[m * n_bins + k] = std::max(filterbank[m * n_bins + k], w);
            bin_weight_sum[k] += w;
        }
    }

    for (float& w : bin_weight_sum) {
        w = std::max(w, 1e-6f);
    }
}

static void mel_to_linear_magnitude(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    int n_bins,
    const std::vector<float>& filterbank,
    const std::vector<float>& bin_weight_sum,
    std::vector<float>& target_mag
) {
    target_mag.assign(n_frames * n_bins, 0.0f);
    for (int t = 0; t < n_frames; t++) {
        for (int k = 0; k < n_bins; k++) {
            float sum = 0.0f;
            for (int m = 0; m < n_mels; m++) {
                const float w = filterbank[m * n_bins + k];
                if (w > 0.0f) {
                    sum += w * mel_value_to_magnitude(mel[m + n_mels * t]);
                }
            }
            target_mag[t * n_bins + k] = sum / bin_weight_sum[k];
        }
    }
}

static void spectrum_from_mag_phase(
    const float* mag,
    const float* phase,
    int n_bins,
    int n_fft,
    std::vector<float>& spectrum
) {
    std::fill(spectrum.begin(), spectrum.end(), 0.0f);
    for (int k = 0; k < n_bins; k++) {
        const float m = mag[k];
        const float p = phase[k];
        spectrum[2 * k] = m * std::cos(p);
        spectrum[2 * k + 1] = m * std::sin(p);
    }
    for (int k = 1; k < n_bins - 1; k++) {
        const int mirror = n_fft - k;
        spectrum[2 * mirror] = spectrum[2 * k];
        spectrum[2 * mirror + 1] = -spectrum[2 * k + 1];
    }
}

static void istft_from_phase(
    const std::vector<float>& target_mag,
    const std::vector<float>& phase,
    int n_frames,
    const GriffinLimConfig& config,
    const std::vector<float>& window,
    bool use_esp_dsp,
    std::vector<float>& audio
) {
    const int n_fft = config.n_fft;
    const int n_bins = n_fft / 2 + 1;
    const int hop = config.hop_length;
    const int internal_len = std::max(n_fft, (n_frames - 1) * hop + n_fft);
    const int out_len = std::max(0, n_frames * hop);

    audio.assign(internal_len, 0.0f);
    std::vector<float> window_norm(internal_len, 0.0f);
    std::vector<float> spectrum(2 * n_fft + kFftGuardFloats, 0.0f);

    for (int t = 0; t < n_frames; t++) {
        spectrum_from_mag_phase(
            target_mag.data() + t * n_bins,
            phase.data() + t * n_bins,
            n_bins,
            n_fft,
            spectrum);
        fft_inplace(spectrum, n_fft, true, use_esp_dsp);

        const int offset = t * hop;
        for (int i = 0; i < n_fft && offset + i < internal_len; i++) {
            const float w = window[i];
            audio[offset + i] += spectrum[2 * i] * w;
            window_norm[offset + i] += w * w;
        }
    }

    for (int i = 0; i < internal_len; i++) {
        if (window_norm[i] > 1e-8f) {
            audio[i] /= window_norm[i];
        }
    }
    if ((int)audio.size() > out_len) {
        audio.resize(out_len);
    }
}

static void update_phase_from_audio(
    const std::vector<float>& audio,
    int n_frames,
    const GriffinLimConfig& config,
    const std::vector<float>& window,
    bool use_esp_dsp,
    std::vector<float>& phase
) {
    const int n_fft = config.n_fft;
    const int n_bins = n_fft / 2 + 1;
    const int hop = config.hop_length;
    std::vector<float> spectrum(2 * n_fft + kFftGuardFloats, 0.0f);

    for (int t = 0; t < n_frames; t++) {
        std::fill(spectrum.begin(), spectrum.end(), 0.0f);
        const int offset = t * hop;
        for (int i = 0; i < n_fft; i++) {
            const int idx = offset + i;
            spectrum[2 * i] = (idx >= 0 && idx < (int)audio.size()) ? audio[idx] * window[i] : 0.0f;
        }

        fft_inplace(spectrum, n_fft, false, use_esp_dsp);
        for (int k = 0; k < n_bins; k++) {
            phase[t * n_bins + k] = std::atan2(spectrum[2 * k + 1], spectrum[2 * k]);
        }
    }
}

static void normalize_audio(std::vector<float>& audio, const GriffinLimConfig& config) {
    if (audio.empty()) {
        return;
    }

    double mean = 0.0;
    for (float v : audio) {
        if (std::isfinite(v)) {
            mean += v;
        }
    }
    mean /= (double)audio.size();

    float peak = 0.0f;
    double rms = 0.0;
    for (float& v : audio) {
        if (!std::isfinite(v)) {
            v = 0.0f;
        } else {
            v -= (float)mean;
        }
        peak = std::max(peak, std::fabs(v));
        rms += (double)v * (double)v;
    }
    rms = std::sqrt(rms / (double)audio.size());

    float gain = 1.0f;
    if (peak > 1e-8f && config.output_peak > 0.0f) {
        gain = config.output_peak / peak;
        gain = std::min(gain, std::max(1.0f, config.max_output_gain));
        for (float& v : audio) {
            v = std::max(-0.95f, std::min(0.95f, v * gain));
        }
    }

    const FloatStats after = compute_stats(audio);
    fprintf(stderr,
            "[GriffinLim] normalize dc=%.6g raw_peak=%.6g raw_rms=%.6g gain=%.3f out_peak=%.6g out_rms=%.6g\n",
            (float)mean,
            peak,
            (float)rms,
            gain,
            after.peak,
            after.rms);
}

} // namespace

std::vector<float> griffin_lim_vocode(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    const GriffinLimConfig& config_in
) {
    GriffinLimConfig config = config_in;
    config.n_mels = n_mels;
    config.n_fft = std::max(64, config.n_fft);
    if (config.n_fft & (config.n_fft - 1)) {
        config.n_fft = 512;
    }
    config.win_length = std::max(1, std::min(config.win_length, config.n_fft));
    config.hop_length = std::max(1, std::min(config.hop_length, config.n_fft));
    config.iterations = std::max(0, config.iterations);

    const uint32_t start_ms = now_ms();
    uint32_t stage_ms = start_ms;
    const int n_bins = config.n_fft / 2 + 1;

#if INFLECT_GL_HAS_ESP_DSP && INFLECT_GRIFFIN_LIM_USE_ESP_DSP
    const bool use_esp_dsp = config.use_esp_dsp_fft &&
                             dsps_fft2r_init_fc32(nullptr, config.n_fft) == ESP_OK;
#else
    const bool use_esp_dsp = false;
#endif

    fprintf(stderr,
            "[GriffinLim] begin frames=%d mels=%d samples_est=%d fft=%d hop=%d iters=%d esp_dsp=%d\n",
            n_frames,
            n_mels,
            n_frames * config.hop_length,
            config.n_fft,
            config.hop_length,
            config.iterations,
            use_esp_dsp ? 1 : 0);

    std::vector<float> filterbank;
    std::vector<float> bin_weight_sum;
    make_mel_filterbank(n_mels, n_bins, config.sample_rate, config.n_fft,
                        config.f_min, config.f_max, filterbank, bin_weight_sum);

    std::vector<float> target_mag;
    mel_to_linear_magnitude(mel, n_mels, n_frames, n_bins,
                            filterbank, bin_weight_sum, target_mag);
    log_stats("mel stats", mel);
    log_stats("magnitude stats", target_mag);

    fprintf(stderr,
            "[GriffinLim] magnitude ready bins=%d stage_ms=%u total_ms=%u\n",
            n_bins,
            (unsigned)(now_ms() - stage_ms),
            (unsigned)(now_ms() - start_ms));
    stage_ms = now_ms();

    std::vector<float> phase(n_frames * n_bins, 0.0f);
    DeterministicRNG rng(config.seed);
    for (float& p : phase) {
        p = (rng.uniform() * 2.0f - 1.0f) * kPi;
    }

    const std::vector<float> window = make_hann_window(config.n_fft, config.win_length);
    std::vector<float> audio;
    for (int iter = 0; iter < config.iterations; iter++) {
        const uint32_t iter_ms = now_ms();
        istft_from_phase(target_mag, phase, n_frames, config, window, use_esp_dsp, audio);
        const uint32_t istft_ms = now_ms() - iter_ms;
        update_phase_from_audio(audio, n_frames, config, window, use_esp_dsp, phase);
        fprintf(stderr,
                "[GriffinLim] iter=%d/%d istft_ms=%u stft_ms=%u total_ms=%u\n",
                iter + 1,
                config.iterations,
                (unsigned)istft_ms,
                (unsigned)(now_ms() - iter_ms - istft_ms),
                (unsigned)(now_ms() - start_ms));
    }

    istft_from_phase(target_mag, phase, n_frames, config, window, use_esp_dsp, audio);
    log_stats("raw audio stats", audio);
    normalize_audio(audio, config);
    fprintf(stderr,
            "[GriffinLim] complete samples=%zu stage_ms=%u total_ms=%u\n",
            audio.size(),
            (unsigned)(now_ms() - stage_ms),
            (unsigned)(now_ms() - start_ms));
    return audio;
}

void griffin_lim_vocode_streaming(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    const GriffinLimConfig& config,
    AudioCallback callback
) {
    std::vector<float> audio = griffin_lim_vocode(mel, n_mels, n_frames, config);
    if (!audio.empty()) {
        callback(audio.data(), audio.size());
    }
}

} // namespace inflect
