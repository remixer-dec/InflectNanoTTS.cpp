#pragma once

#include "inflect-nano.h"
#include "../ggml/include/ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>

namespace inflect {

// One pool for the entire utterance, not one pthread create/join per window.
// Idle workers sleep immediately instead of using GGML's desktop spin budget.
class SanoThreadpoolScope {
public:
    SanoThreadpoolScope(ggml_backend_t backend, int threads) : backend_(backend) {
        if (!backend_ || !ggml_backend_is_cpu(backend_)) return;
        auto params = ggml_threadpool_params_default(threads);
        params.poll = 0;
        params.paused = true;
        pool_ = ggml_threadpool_new(&params);
        ggml_backend_cpu_set_threadpool(backend_, pool_);
    }
    ~SanoThreadpoolScope() {
        if (!pool_) return;
        ggml_backend_cpu_set_threadpool(backend_, nullptr);
        ggml_threadpool_free(pool_);
    }
    SanoThreadpoolScope(const SanoThreadpoolScope&) = delete;
    SanoThreadpoolScope& operator=(const SanoThreadpoolScope&) = delete;
private:
    ggml_backend_t backend_;
    ggml_threadpool_t pool_ = nullptr;
};

// Large activations must not depend on the platform's default new/malloc
// routing. Allocation failure is recoverable, including on exceptionless MCUs.
class SanoBuffer {
public:
    SanoBuffer() = default;
    ~SanoBuffer() { runtime_free_scratch(data_); }
    SanoBuffer(const SanoBuffer&) = delete;
    SanoBuffer& operator=(const SanoBuffer&) = delete;
    SanoBuffer(SanoBuffer&& other) noexcept { swap(other); }
    SanoBuffer& operator=(SanoBuffer&& other) noexcept {
        SanoBuffer moved(std::move(other));
        swap(moved);
        return *this;
    }
    bool allocate(size_t count) {
        SanoBuffer next;
        if (count > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
        if (count) {
            next.data_ = static_cast<float*>(runtime_alloc_scratch(
                count * sizeof(float), ScratchMemoryKind::Psram));
            if (!next.data_) {
                std::fprintf(stderr, "[SanoMemory] allocation failed bytes=%zu\n",
                             count * sizeof(float));
                runtime_trace_heap("sano allocation failed");
                return false;
            }
        }
        next.size_ = count;
        swap(next);
        return true;
    }
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
private:
    void swap(SanoBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }
    float* data_ = nullptr;
    size_t size_ = 0;
};

// Clip at real utterance boundaries: synthesizing artificial zero frames
// outside them would introduce convolution biases and change edge samples.
template<class Run>
SanoBuffer sano_run_tiled(const float* input, int frames, int in_channels,
                          int out_channels, int rate, int core, int halo,
                          const char* label, Run run) {
    const uint32_t started = runtime_now_ms();
    SanoBuffer output;
    if (!input || frames <= 0 || rate <= 0 || core <= 0 || halo < 0 ||
        frames > std::numeric_limits<int>::max() / rate ||
        !output.allocate(static_cast<size_t>(frames) * rate * out_channels)) {
        return {};
    }
    int chunks = 0;
    uint32_t last_report = started;
    for (int start = 0; start < frames;) {
#if defined(INFLECT_LOW_MEMORY)
        runtime_cooperate();
#endif
        if (runtime_cancelled()) return {};
        int end = start + std::min(core, frames - start);
        const int left = std::max(0, start - halo);
        const int right = end + std::min(halo, frames - end);
        // The graph already sees the true right boundary. Its remaining halo
        // outputs are valid, so keep them instead of recomputing a tiny tail.
        // Do not enlarge the graph or shorten the left dependency halo.
        if (right == frames) end = frames;
        SanoBuffer tile = run(input, frames, left, right - left);
        const int tile_frames = (right - left) * rate;
        if (tile.size() != static_cast<size_t>(tile_frames) * out_channels) return {};
        for (int channel = 0; channel < out_channels; ++channel) {
            std::copy_n(tile.data() + static_cast<size_t>(channel) * tile_frames +
                            (start - left) * rate,
                        (end - start) * rate,
                        output.data() + static_cast<size_t>(channel) * frames * rate +
                            start * rate);
        }
        start = end;
        ++chunks;
        const uint32_t now = runtime_now_ms();
        if (now - last_report >= 1000) {
            std::fprintf(stderr, "[SanoProgress] %s frames=%d/%d elapsed_ms=%u\n",
                         label, start, frames, static_cast<unsigned>(now - started));
            last_report = now;
        }
    }
    std::fprintf(stderr,
                 "[SanoStage] %s frames=%d channels=%d->%d chunks=%d "
                 "core=%d halo=%d output_bytes=%zu elapsed_ms=%u\n",
                 label, frames, in_channels, out_channels, chunks, core, halo,
                 output.size() * sizeof(float),
                 static_cast<unsigned>(runtime_now_ms() - started));
    return output;
}

} // namespace inflect
