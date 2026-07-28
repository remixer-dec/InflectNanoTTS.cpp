#include "vocoder_model.h"
#include "inflect-nano.h"
#include "memory_trace.h"
#include "vocoder_quant_math.h"
#include "../ggml/include/ggml-cpu.h"
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>

#if __has_include(<esp_dsp.h>)
#include <esp_dsp.h>
#define INFLECT_HAS_ESP_DSP 1
#else
#define INFLECT_HAS_ESP_DSP 0
#endif

#ifndef INFLECT_USE_ESP_DSP_CONTIG
#define INFLECT_USE_ESP_DSP_CONTIG 0
#endif

#ifndef INFLECT_USE_ESP_DSP_STRIDED
#define INFLECT_USE_ESP_DSP_STRIDED 1
#endif

#ifndef INFLECT_PROFILE_VOCODER_OPS
#define INFLECT_PROFILE_VOCODER_OPS 0
#endif

namespace inflect {

struct VocoderFloatScratch {
    float* ptr = nullptr;
    size_t cap = 0;

    ~VocoderFloatScratch() {
        runtime_free_scratch(ptr);
    }

#if defined(INFLECT_LOW_MEMORY)
    bool try_resize(size_t n) {
        if (n <= cap) {
            return true;
        }
        runtime_free_scratch(ptr);
        ptr = nullptr;
        cap = 0;
        float* next = static_cast<float*>(
            runtime_alloc_scratch(n * sizeof(float), ScratchMemoryKind::Psram));
        if (next == nullptr) {
            return false;
        }
        ptr = next;
        cap = n;
        return true;
    }
#endif

    void resize(size_t n) {
#if defined(INFLECT_LOW_MEMORY)
        if (!try_resize(n)) {
            fprintf(stderr, "[VocoderModel] scratch allocation failed floats=%zu\n", n);
            std::abort();
        }
#else
        if (n <= cap) {
            return;
        }
        float* next = static_cast<float*>(
            runtime_alloc_scratch(n * sizeof(float), ScratchMemoryKind::Psram));
        if (next == nullptr) {
            fprintf(stderr, "[VocoderModel] scratch allocation failed floats=%zu\n", n);
            std::abort();
        }
        runtime_free_scratch(ptr);
        ptr = next;
        cap = n;
#endif
    }

    float* data() { return ptr; }
    const float* data() const { return ptr; }
};

struct VocoderInternalFloatScratch {
    float* ptr = nullptr;
    size_t cap = 0;

    ~VocoderInternalFloatScratch() {
        runtime_free_scratch(ptr);
    }

    void resize(size_t n) {
        if (n <= cap) {
            return;
        }
        float* next = static_cast<float*>(
            runtime_alloc_scratch(n * sizeof(float), ScratchMemoryKind::InternalPreferred));
        if (next == nullptr) {
            fprintf(stderr, "[VocoderModel] internal scratch allocation failed floats=%zu\n", n);
            std::abort();
        }
        runtime_free_scratch(ptr);
        ptr = next;
        cap = n;
    }

    float* data() { return ptr; }
};

#if defined(INFLECT_LOW_MEMORY)
constexpr int kQ4BlockElements = 32;

struct PackedQ4Block {
    ggml_fp16_t scale;
    uint8_t quants[kQ4BlockElements / 2];
};

static_assert(
    sizeof(PackedQ4Block) ==
        sizeof(ggml_fp16_t) + kQ4BlockElements / 2,
    "unexpected Q4_0 block layout");

using Q4Q8Scale = float;

static Q4Q8Scale cache_weight_scale(ggml_fp16_t scale) {
    return ggml_fp16_to_fp32(scale);
}

static float read_cached_scale(Q4Q8Scale scale) {
    return scale;
}

struct VocoderInternalByteScratch {
    void* allocation = nullptr;
    void* ptr = nullptr;
    size_t cap = 0;

    ~VocoderInternalByteScratch() {
        runtime_free_scratch(allocation);
    }

    bool try_resize(size_t n) {
        if (n <= cap) {
            return true;
        }
        runtime_free_scratch(allocation);
        allocation = runtime_alloc_scratch(
            n + 15, ScratchMemoryKind::InternalPreferred);
        if (allocation == nullptr) {
            ptr = nullptr;
            cap = 0;
            return false;
        }
        const uintptr_t address =
            reinterpret_cast<uintptr_t>(allocation);
        ptr = reinterpret_cast<void*>(
            (address + 15) & ~uintptr_t(15));
        cap = n;
        return true;
    }

    void* data() { return ptr; }
    const void* data() const { return ptr; }
};
#endif

static uint32_t now_ms() { return runtime_now_ms(); }

#if INFLECT_PROFILE_VOCODER_OPS
struct VocodeOpProfile {
    std::atomic<uint32_t> conv1d_calls{0};
    std::atomic<uint32_t> conv_transpose_calls{0};
    std::atomic<uint64_t> conv1d_ms{0};
    std::atomic<uint64_t> conv_transpose_ms{0};
    std::atomic<uint64_t> conv1d_outputs{0};
    std::atomic<uint64_t> conv_transpose_outputs{0};
    std::atomic<uint64_t> conv1d_macs{0};
    std::atomic<uint64_t> conv_transpose_macs{0};
};

static VocodeOpProfile g_vocode_profile;

struct VocodePackedProfile {
    std::atomic<uint64_t> input_gather_cycles{0};
    std::atomic<uint64_t> reused_input_blocks{0};
    std::atomic<uint64_t> input_quant_cycles{0};
    std::atomic<uint64_t> input_max_cycles{0};
    std::atomic<uint64_t> input_scale_cycles{0};
    std::atomic<uint64_t> input_convert_cycles{0};
    std::atomic<uint64_t> q4_unpack_cycles{0};
    std::atomic<uint64_t> s8_dot_cycles{0};
    std::atomic<uint64_t> scale_reduce_cycles{0};
    std::atomic<uint64_t> output_write_cycles{0};
};

static VocodePackedProfile g_vocode_packed_profile;

struct VocodeOpProfileBucket {
    const char* label;
    std::atomic<uint32_t> calls;
    std::atomic<uint64_t> elapsed_ms;
    std::atomic<uint64_t> outputs;
    std::atomic<uint64_t> macs;
    std::atomic<uint64_t> input_blocks;
    std::atomic<uint64_t> zero_input_blocks;
    std::atomic<uint64_t> input_values;
    std::atomic<uint64_t> zero_input_values;
};

static VocodeOpProfileBucket g_vocode_profile_buckets[] = {
    {"conv_pre", {}, {}, {}, {}, {}, {}, {}, {}},
    {"upsample", {}, {}, {}, {}, {}, {}, {}, {}},
    {"resblock.convs1", {}, {}, {}, {}, {}, {}, {}, {}},
    {"resblock.convs2", {}, {}, {}, {}, {}, {}, {}, {}},
    {"conv_post", {}, {}, {}, {}, {}, {}, {}, {}},
};

static void vocode_profile_reset() {
    g_vocode_profile.conv1d_calls.store(0);
    g_vocode_profile.conv_transpose_calls.store(0);
    g_vocode_profile.conv1d_ms.store(0);
    g_vocode_profile.conv_transpose_ms.store(0);
    g_vocode_profile.conv1d_outputs.store(0);
    g_vocode_profile.conv_transpose_outputs.store(0);
    g_vocode_profile.conv1d_macs.store(0);
    g_vocode_profile.conv_transpose_macs.store(0);
    g_vocode_packed_profile.input_quant_cycles.store(0);
    g_vocode_packed_profile.input_gather_cycles.store(0);
    g_vocode_packed_profile.reused_input_blocks.store(0);
    g_vocode_packed_profile.input_max_cycles.store(0);
    g_vocode_packed_profile.input_scale_cycles.store(0);
    g_vocode_packed_profile.input_convert_cycles.store(0);
    g_vocode_packed_profile.q4_unpack_cycles.store(0);
    g_vocode_packed_profile.s8_dot_cycles.store(0);
    g_vocode_packed_profile.scale_reduce_cycles.store(0);
    g_vocode_packed_profile.output_write_cycles.store(0);
    for (auto& bucket : g_vocode_profile_buckets) {
        bucket.calls.store(0);
        bucket.elapsed_ms.store(0);
        bucket.outputs.store(0);
        bucket.macs.store(0);
        bucket.input_blocks.store(0);
        bucket.zero_input_blocks.store(0);
        bucket.input_values.store(0);
        bucket.zero_input_values.store(0);
    }
}

static void vocode_profile_add_packed(
    uint64_t input_gather_cycles,
    uint64_t reused_input_blocks,
    uint64_t input_quant_cycles,
    uint64_t input_max_cycles,
    uint64_t input_scale_cycles,
    uint64_t input_convert_cycles,
    uint64_t q4_unpack_cycles,
    uint64_t s8_dot_cycles,
    uint64_t scale_reduce_cycles,
    uint64_t output_write_cycles
) {
    g_vocode_packed_profile.input_gather_cycles.fetch_add(
        input_gather_cycles);
    g_vocode_packed_profile.reused_input_blocks.fetch_add(
        reused_input_blocks);
    g_vocode_packed_profile.input_quant_cycles.fetch_add(
        input_quant_cycles);
    g_vocode_packed_profile.input_max_cycles.fetch_add(
        input_max_cycles);
    g_vocode_packed_profile.input_scale_cycles.fetch_add(
        input_scale_cycles);
    g_vocode_packed_profile.input_convert_cycles.fetch_add(
        input_convert_cycles);
    g_vocode_packed_profile.q4_unpack_cycles.fetch_add(
        q4_unpack_cycles);
    g_vocode_packed_profile.s8_dot_cycles.fetch_add(
        s8_dot_cycles);
    g_vocode_packed_profile.scale_reduce_cycles.fetch_add(
        scale_reduce_cycles);
    g_vocode_packed_profile.output_write_cycles.fetch_add(
        output_write_cycles);
}

static void vocode_profile_add_input_sparsity(
    const char* label,
    uint64_t input_blocks,
    uint64_t zero_input_blocks,
    uint64_t input_values,
    uint64_t zero_input_values
) {
    if (label == nullptr) {
        return;
    }
    for (auto& bucket : g_vocode_profile_buckets) {
        if (std::strcmp(bucket.label, label) == 0) {
            bucket.input_blocks.fetch_add(input_blocks);
            bucket.zero_input_blocks.fetch_add(
                zero_input_blocks);
            bucket.input_values.fetch_add(input_values);
            bucket.zero_input_values.fetch_add(
                zero_input_values);
            return;
        }
    }
}

static void vocode_profile_add_bucket(const char* label, uint32_t elapsed_ms, uint64_t outputs, uint64_t macs) {
    if (label == nullptr) {
        return;
    }
    for (auto& bucket : g_vocode_profile_buckets) {
        if (std::strcmp(bucket.label, label) == 0) {
            bucket.calls.fetch_add(1);
            bucket.elapsed_ms.fetch_add(elapsed_ms);
            bucket.outputs.fetch_add(outputs);
            bucket.macs.fetch_add(macs);
            return;
        }
    }
}

static void vocode_profile_add_conv1d(const char* label, uint32_t elapsed_ms, uint64_t outputs, uint64_t macs) {
    g_vocode_profile.conv1d_calls.fetch_add(1);
    g_vocode_profile.conv1d_ms.fetch_add(elapsed_ms);
    g_vocode_profile.conv1d_outputs.fetch_add(outputs);
    g_vocode_profile.conv1d_macs.fetch_add(macs);
    vocode_profile_add_bucket(label, elapsed_ms, outputs, macs);
}

static void vocode_profile_add_conv_transpose(const char* label, uint32_t elapsed_ms, uint64_t outputs, uint64_t macs) {
    g_vocode_profile.conv_transpose_calls.fetch_add(1);
    g_vocode_profile.conv_transpose_ms.fetch_add(elapsed_ms);
    g_vocode_profile.conv_transpose_outputs.fetch_add(outputs);
    g_vocode_profile.conv_transpose_macs.fetch_add(macs);
    vocode_profile_add_bucket(label, elapsed_ms, outputs, macs);
}

static void vocode_profile_log(uint32_t graph_compute_ms) {
    const uint64_t conv1d_ms = g_vocode_profile.conv1d_ms.load();
    const uint64_t conv_transpose_ms = g_vocode_profile.conv_transpose_ms.load();
    const uint64_t custom_worker_ms =
        conv1d_ms + conv_transpose_ms;
    const uint64_t active_workers_x100 =
        graph_compute_ms > 0
            ? custom_worker_ms * 100ULL / graph_compute_ms
            : 0;
    fprintf(stderr,
            "[VocoderProfile] graph_ms=%u custom_worker_ms=%llu "
            "active_workers_x100=%llu "
            "conv1d_calls=%u conv1d_ms=%llu conv1d_outputs=%llu conv1d_macs=%llu "
            "convT_calls=%u convT_ms=%llu convT_outputs=%llu convT_macs=%llu\n",
            (unsigned)graph_compute_ms,
            (unsigned long long)custom_worker_ms,
            (unsigned long long)active_workers_x100,
            (unsigned)g_vocode_profile.conv1d_calls.load(),
            (unsigned long long)conv1d_ms,
            (unsigned long long)g_vocode_profile.conv1d_outputs.load(),
            (unsigned long long)g_vocode_profile.conv1d_macs.load(),
            (unsigned)g_vocode_profile.conv_transpose_calls.load(),
            (unsigned long long)conv_transpose_ms,
            (unsigned long long)g_vocode_profile.conv_transpose_outputs.load(),
            (unsigned long long)g_vocode_profile.conv_transpose_macs.load());
    for (const auto& bucket : g_vocode_profile_buckets) {
        const uint32_t calls = bucket.calls.load();
        if (calls == 0) {
            continue;
        }
        const uint64_t elapsed_ms = bucket.elapsed_ms.load();
        const uint64_t outputs = bucket.outputs.load();
        const uint64_t macs = bucket.macs.load();
        fprintf(stderr,
                "[VocoderProfile] op=%s calls=%u ms=%llu outputs=%llu macs=%llu us_per_output=%llu ns_per_mac=%llu\n",
                bucket.label,
                (unsigned)calls,
                (unsigned long long)elapsed_ms,
                (unsigned long long)outputs,
                (unsigned long long)macs,
                outputs > 0 ? (unsigned long long)((elapsed_ms * 1000ULL) / outputs) : 0ULL,
                macs > 0 ? (unsigned long long)((elapsed_ms * 1000000ULL) / macs) : 0ULL);
        const uint64_t input_blocks =
            bucket.input_blocks.load();
        const uint64_t zero_input_blocks =
            bucket.zero_input_blocks.load();
        const uint64_t input_values =
            bucket.input_values.load();
        const uint64_t zero_input_values =
            bucket.zero_input_values.load();
        if (input_blocks > 0) {
            fprintf(
                stderr,
                "[VocoderInputSparsity] op=%s "
                "blocks=%llu zero_blocks=%llu "
                "zero_blocks_pct_x100=%llu "
                "values=%llu zero_values=%llu "
                "zero_values_pct_x100=%llu\n",
                bucket.label,
                (unsigned long long)input_blocks,
                (unsigned long long)zero_input_blocks,
                (unsigned long long)(
                    zero_input_blocks * 10000ULL /
                    input_blocks),
                (unsigned long long)input_values,
                (unsigned long long)zero_input_values,
                (unsigned long long)(
                    input_values > 0
                        ? zero_input_values * 10000ULL /
                              input_values
                        : 0));
        }
    }
#if defined(INFLECT_LOW_MEMORY)
    fprintf(stderr,
            "[VocoderPackedProfile] tile=%d input_gather_cycles=%llu "
            "reused_input_blocks=%llu "
            "input_quant_cycles=%llu "
            "input_max_cycles=%llu input_scale_cycles=%llu "
            "input_convert_cycles=%llu "
            "q4_unpack_cycles=%llu s8_dot_cycles=%llu "
            "scale_reduce_cycles=%llu output_write_cycles=%llu\n",
            runtime_packed_quant_time_tile(),
            (unsigned long long)
                g_vocode_packed_profile.input_gather_cycles.load(),
            (unsigned long long)
                g_vocode_packed_profile.reused_input_blocks.load(),
            (unsigned long long)
                g_vocode_packed_profile.input_quant_cycles.load(),
            (unsigned long long)
                g_vocode_packed_profile.input_max_cycles.load(),
            (unsigned long long)
                g_vocode_packed_profile.input_scale_cycles.load(),
            (unsigned long long)
                g_vocode_packed_profile.input_convert_cycles.load(),
            (unsigned long long)
                g_vocode_packed_profile.q4_unpack_cycles.load(),
            (unsigned long long)
                g_vocode_packed_profile.s8_dot_cycles.load(),
            (unsigned long long)
                g_vocode_packed_profile.scale_reduce_cycles.load(),
            (unsigned long long)
                g_vocode_packed_profile.output_write_cycles.load());
#endif
}
#else
static void vocode_profile_reset() {}
static void vocode_profile_add_conv1d(const char*, uint32_t, uint64_t, uint64_t) {}
static void vocode_profile_add_conv_transpose(const char*, uint32_t, uint64_t, uint64_t) {}
static void vocode_profile_log(uint32_t) {}
#endif

static ggml_tensor* channel_param_3d(ggml_context* gctx, ggml_tensor* t) {
    return ggml_reshape_3d(gctx, t, 1, t->ne[0], 1);
}

static float dot_f32_strided(const float* a, int step_a, const float* b, int step_b, int n) {
    float out = 0.0f;
    if (n <= 0) {
        return 0.0f;
    }
#if INFLECT_HAS_ESP_DSP && INFLECT_USE_ESP_DSP_CONTIG
    if (step_a == 1 && step_b == 1) {
        if (dsps_dotprod_f32(a, b, &out, n) == ESP_OK) {
            return out;
        }
    }
#endif
#if INFLECT_HAS_ESP_DSP && INFLECT_USE_ESP_DSP_STRIDED
    if (step_a > 0 && step_b > 0) {
        out = 0.0f;
        if (dsps_dotprode_f32(a, b, &out, n, step_a, step_b) == ESP_OK) {
            return out;
        }
    }
    out = 0.0f;
#endif
    int i = 0;
    for (; i + 3 < n; i += 4) {
        out += a[i * step_a] * b[i * step_b] +
               a[(i + 1) * step_a] * b[(i + 1) * step_b] +
               a[(i + 2) * step_a] * b[(i + 2) * step_b] +
               a[(i + 3) * step_a] * b[(i + 3) * step_b];
    }
    for (; i < n; i++) {
        out += a[i * step_a] * b[i * step_b];
    }
    return out;
}

static float tensor_get_f32(const ggml_tensor* t, int64_t i0, int64_t i1 = 0, int64_t i2 = 0) {
    const char* ptr = (const char*)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2];
    switch (t->type) {
        case GGML_TYPE_F32:
            return *(const float*)ptr;
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t*)ptr);
        default:
            break;
    }
    if (ggml_is_quantized(t->type)) {
        const auto* traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            fprintf(stderr, "[VocoderModel] unsupported quantized tensor read type %s for %s\n",
                    ggml_type_name(t->type), t->name);
            std::abort();
        }

        const char* row_ptr = (const char*)t->data + i1 * t->nb[1] + i2 * t->nb[2];
        struct QuantRowCache {
            const ggml_tensor* tensor = nullptr;
            const char* row = nullptr;
            std::vector<float> values;
        };
        thread_local QuantRowCache cache;
        if (cache.tensor != t || cache.row != row_ptr || (int64_t)cache.values.size() != t->ne[0]) {
            cache.tensor = t;
            cache.row = row_ptr;
            cache.values.resize(t->ne[0]);
            traits->to_float(row_ptr, cache.values.data(), t->ne[0]);
        }
        return cache.values[i0];
    }
    fprintf(stderr, "[VocoderModel] unsupported direct tensor read type %s for %s\n",
            ggml_type_name(t->type), t->name);
    std::abort();
}

static void tensor_set_f32(ggml_tensor* t, float v, int64_t i0, int64_t i1, int64_t i2) {
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "[VocoderModel] unsupported direct tensor write type %s for %s\n",
                ggml_type_name(t->type), t->name);
        std::abort();
    }
    *(float*)((char*)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2]) = v;
}

static bool is_resblock_conv_label(const char* label) {
    return label != nullptr &&
           (std::strcmp(label, "resblock.convs1") == 0 ||
            std::strcmp(label, "resblock.convs2") == 0);
}

#if defined(INFLECT_LOW_MEMORY)
struct Q4Q8TileDot {
    int32_t elements = 0;
    int32_t blocks = 0;
    bool sparse_input_blocks = false;
    bool elide_zero_input_values = false;
    VocoderInternalByteScratch input_values;
    VocoderInternalByteScratch input_scales;
    VocoderInternalByteScratch weight_values;
    VocoderInternalByteScratch weight_scale_bits;
    VocoderInternalByteScratch weight_scales;
    VocoderInternalByteScratch sums;
    VocoderInternalByteScratch results;
    QuantizeF32ToQ8Blocks32Fn quantize_blocks_32 = nullptr;
#if INFLECT_PROFILE_VOCODER_OPS
    uint64_t input_gather_cycles = 0;
    uint64_t reused_input_blocks = 0;
    uint64_t input_quant_cycles = 0;
    uint64_t input_max_cycles = 0;
    uint64_t input_scale_cycles = 0;
    uint64_t input_convert_cycles = 0;
    uint64_t q4_unpack_cycles = 0;
    uint64_t s8_dot_cycles = 0;
    uint64_t scale_reduce_cycles = 0;
    uint64_t profiled_input_blocks = 0;
    uint64_t profiled_zero_input_blocks = 0;
    uint64_t profiled_input_values = 0;
    uint64_t profiled_zero_input_values = 0;
#endif

    bool init(
        int64_t count,
        int tile_capacity,
        bool sparse_inputs = false
    ) {
        if (count <= 0 || count % kQ4BlockElements != 0 ||
            count > INT32_MAX || tile_capacity <= 0) {
            return false;
        }
        elements = static_cast<int32_t>(count);
        blocks = elements / kQ4BlockElements;
        sparse_input_blocks = sparse_inputs;
        elide_zero_input_values =
            sparse_inputs &&
            runtime_has_s8_scaled_dot_blocks_32();
        quantize_blocks_32 =
            runtime_config().quantize_f32_to_q8_blocks_32;
        const size_t input_value_bytes =
            static_cast<size_t>(elements) *
            static_cast<size_t>(tile_capacity);
        const bool allocated =
               input_values.try_resize(input_value_bytes) &&
               input_scales.try_resize(
                   static_cast<size_t>(blocks) *
                   static_cast<size_t>(tile_capacity) *
                   sizeof(Q4Q8Scale)) &&
               weight_values.try_resize(
                   static_cast<size_t>(elements)) &&
               weight_scale_bits.try_resize(
                   static_cast<size_t>(blocks) *
                   sizeof(uint16_t)) &&
               weight_scales.try_resize(
                   static_cast<size_t>(blocks) *
                   sizeof(Q4Q8Scale)) &&
               results.try_resize(
                   static_cast<size_t>(tile_capacity) *
                   sizeof(float)) &&
               (runtime_has_s8_scaled_dot_blocks_32() ||
                sums.try_resize(
                    static_cast<size_t>(blocks) *
                    static_cast<size_t>(tile_capacity) *
                    sizeof(int32_t)));
        return allocated;
    }

    __attribute__((always_inline)) inline void quantize_blocks(
        int tile,
        int32_t first_block,
        const float* values,
        int32_t block_count,
        bool skip_zero_blocks
    ) {
#if INFLECT_PROFILE_VOCODER_OPS
        uint64_t window_zero_values = 0;
        uint64_t window_zero_blocks = 0;
        for (int32_t block = 0; block < block_count; ++block) {
            uint64_t block_zero_values = 0;
            for (int index = 0; index < kQ4BlockElements; ++index) {
                block_zero_values += static_cast<uint64_t>(
                    values[block * kQ4BlockElements + index] ==
                    0.0f);
            }
            window_zero_values += block_zero_values;
            window_zero_blocks += static_cast<uint64_t>(
                block_zero_values == kQ4BlockElements);
        }
        profiled_input_values +=
            static_cast<uint64_t>(
                block_count * kQ4BlockElements);
        profiled_zero_input_values += window_zero_values;
        profiled_input_blocks +=
            static_cast<uint64_t>(block_count);
        profiled_zero_input_blocks += window_zero_blocks;
        const uint32_t started = runtime_now_cycles();
#endif
        auto* quantized =
            static_cast<int8_t*>(input_values.data()) +
            static_cast<size_t>(tile) *
                static_cast<size_t>(elements) +
            static_cast<size_t>(first_block) *
                kQ4BlockElements;
        auto* scales =
            static_cast<Q4Q8Scale*>(input_scales.data()) +
            static_cast<size_t>(tile) *
                static_cast<size_t>(blocks) +
            static_cast<size_t>(first_block);
#if INFLECT_PROFILE_VOCODER_OPS
        uint64_t max_cycles = 0;
        uint64_t scale_cycles = 0;
        uint64_t convert_cycles = 0;
        uint64_t* max_cycles_ptr = &max_cycles;
        uint64_t* scale_cycles_ptr = &scale_cycles;
        uint64_t* convert_cycles_ptr = &convert_cycles;
#else
        uint64_t* max_cycles_ptr = nullptr;
        uint64_t* scale_cycles_ptr = nullptr;
        uint64_t* convert_cycles_ptr = nullptr;
#endif
        if (quantize_blocks_32 != nullptr) {
            quantize_blocks_32(
                values,
                quantized,
                scales,
                static_cast<size_t>(block_count),
                skip_zero_blocks,
                max_cycles_ptr,
                scale_cycles_ptr,
                convert_cycles_ptr);
        } else {
            runtime_quantize_f32_to_q8_blocks_32(
                values,
                quantized,
                scales,
                static_cast<size_t>(block_count),
                skip_zero_blocks,
                max_cycles_ptr,
                scale_cycles_ptr,
                convert_cycles_ptr);
        }
#if INFLECT_PROFILE_VOCODER_OPS
        input_max_cycles += max_cycles;
        input_scale_cycles += scale_cycles;
        input_convert_cycles += convert_cycles;
        input_quant_cycles += static_cast<uint32_t>(
            runtime_now_cycles() - started);
#endif
    }

    __attribute__((always_inline)) inline void store_zero_blocks(
        int tile,
        int32_t first_block,
        int32_t block_count
    ) {
        auto* quantized =
            static_cast<int8_t*>(input_values.data()) +
            static_cast<size_t>(tile) *
                static_cast<size_t>(elements) +
            static_cast<size_t>(first_block) *
                kQ4BlockElements;
        auto* scales =
            static_cast<Q4Q8Scale*>(input_scales.data()) +
            static_cast<size_t>(tile) *
                static_cast<size_t>(blocks) +
            static_cast<size_t>(first_block);
#if INFLECT_PROFILE_VOCODER_OPS
        const uint32_t started = runtime_now_cycles();
        profiled_input_blocks +=
            static_cast<uint64_t>(block_count);
        profiled_zero_input_blocks +=
            static_cast<uint64_t>(block_count);
        profiled_input_values += static_cast<uint64_t>(
            block_count * kQ4BlockElements);
        profiled_zero_input_values += static_cast<uint64_t>(
            block_count * kQ4BlockElements);
#endif
        // The sparse scaled-dot kernel checks the scale before touching Q8
        // values, so a zero block needs only one zero scale store. Keep the
        // byte write for the portable non-sparse fallback.
        if (!elide_zero_input_values) {
            runtime_store_zero_s8_blocks_32(
                quantized,
                static_cast<size_t>(block_count));
        }
        std::fill(
            scales,
            scales + block_count,
            0.0f);
#if INFLECT_PROFILE_VOCODER_OPS
        input_quant_cycles += static_cast<uint32_t>(
            runtime_now_cycles() - started);
#endif
    }

    __attribute__((always_inline)) inline void copy_blocks(
        int destination_tile,
        int32_t destination_first_block,
        int source_tile,
        int32_t source_first_block,
        int32_t block_count
#if INFLECT_PROFILE_VOCODER_OPS
        ,
        uint64_t source_zero_blocks,
        uint64_t source_zero_values
#endif
    ) {
        auto* destination_values =
            static_cast<int8_t*>(input_values.data()) +
            static_cast<size_t>(destination_tile) *
                static_cast<size_t>(elements) +
            static_cast<size_t>(destination_first_block) *
                kQ4BlockElements;
        const auto* source_values =
            static_cast<const int8_t*>(input_values.data()) +
            static_cast<size_t>(source_tile) *
                static_cast<size_t>(elements) +
            static_cast<size_t>(source_first_block) *
                kQ4BlockElements;
        auto* destination_scales =
            static_cast<Q4Q8Scale*>(input_scales.data()) +
            static_cast<size_t>(destination_tile) *
                static_cast<size_t>(blocks) +
            static_cast<size_t>(destination_first_block);
        const auto* source_scales =
            static_cast<const Q4Q8Scale*>(input_scales.data()) +
            static_cast<size_t>(source_tile) *
                static_cast<size_t>(blocks) +
            static_cast<size_t>(source_first_block);
        std::memcpy(
            destination_values,
            source_values,
            static_cast<size_t>(block_count) *
                kQ4BlockElements);
        std::memcpy(
            destination_scales,
            source_scales,
            static_cast<size_t>(block_count) *
                sizeof(Q4Q8Scale));
#if INFLECT_PROFILE_VOCODER_OPS
        reused_input_blocks +=
            static_cast<uint64_t>(block_count);
        profiled_input_blocks +=
            static_cast<uint64_t>(block_count);
        profiled_zero_input_blocks += source_zero_blocks;
        profiled_input_values += static_cast<uint64_t>(
            block_count * kQ4BlockElements);
        profiled_zero_input_values += source_zero_values;
#endif
    }

    void unpack_weight(const void* row) {
#if INFLECT_PROFILE_VOCODER_OPS
        const uint32_t started = runtime_now_cycles();
#endif
        auto* values =
            static_cast<int8_t*>(weight_values.data());
        auto* scales =
            static_cast<Q4Q8Scale*>(
                weight_scales.data());
        auto* scale_bits =
            static_cast<uint16_t*>(
                weight_scale_bits.data());
        runtime_unpack_q4_0_blocks_32(
            static_cast<const uint8_t*>(row),
            sizeof(PackedQ4Block),
            values,
            scale_bits,
            static_cast<size_t>(blocks));
        for (int32_t block = 0; block < blocks; ++block) {
            scales[block] =
                cache_weight_scale(scale_bits[block]);
        }
#if INFLECT_PROFILE_VOCODER_OPS
        q4_unpack_cycles += static_cast<uint32_t>(
            runtime_now_cycles() - started);
#endif
    }

    void calculate(int rows) {
        if (runtime_has_s8_scaled_dot_blocks_32()) {
#if INFLECT_PROFILE_VOCODER_OPS
            uint64_t dot_cycles = 0;
            uint64_t reduce_cycles = 0;
            uint64_t* dot_cycles_ptr = &dot_cycles;
            uint64_t* reduce_cycles_ptr = &reduce_cycles;
#else
            uint64_t* dot_cycles_ptr = nullptr;
            uint64_t* reduce_cycles_ptr = nullptr;
#endif
            runtime_dot_s8_scaled_blocks_32(
                static_cast<const int8_t*>(
                    weight_values.data()),
                static_cast<const Q4Q8Scale*>(
                    weight_scales.data()),
                static_cast<const int8_t*>(
                    input_values.data()),
                static_cast<const Q4Q8Scale*>(
                    input_scales.data()),
                static_cast<float*>(results.data()),
                static_cast<size_t>(blocks),
                static_cast<size_t>(rows),
                sparse_input_blocks,
                dot_cycles_ptr,
                reduce_cycles_ptr);
#if INFLECT_PROFILE_VOCODER_OPS
            s8_dot_cycles += dot_cycles;
            scale_reduce_cycles += reduce_cycles;
#endif
            return;
        }
#if INFLECT_PROFILE_VOCODER_OPS
        const uint32_t dot_started = runtime_now_cycles();
#endif
        runtime_dot_s8_blocks_32(
            static_cast<const int8_t*>(
                weight_values.data()),
            static_cast<const int8_t*>(
                input_values.data()),
            static_cast<int32_t*>(sums.data()),
            static_cast<size_t>(blocks),
            static_cast<size_t>(rows));
#if INFLECT_PROFILE_VOCODER_OPS
        s8_dot_cycles += static_cast<uint32_t>(
            runtime_now_cycles() - dot_started);
        const uint32_t reduce_started = runtime_now_cycles();
#endif
        auto* output = static_cast<float*>(results.data());
        const auto* products =
            static_cast<const int32_t*>(sums.data());
        const auto* input_scale =
            static_cast<const Q4Q8Scale*>(
                input_scales.data());
        const auto* weight_scale =
            static_cast<const Q4Q8Scale*>(
                weight_scales.data());
        for (int row = 0; row < rows; ++row) {
            float result = 0.0f;
            for (int32_t block = 0; block < blocks; ++block) {
                const size_t index =
                    static_cast<size_t>(row) *
                        static_cast<size_t>(blocks) +
                    static_cast<size_t>(block);
                result +=
                    static_cast<float>(products[index]) *
                    read_cached_scale(weight_scale[block]) *
                    read_cached_scale(input_scale[index]);
            }
            output[row] = result;
        }
#if INFLECT_PROFILE_VOCODER_OPS
        scale_reduce_cycles += static_cast<uint32_t>(
            runtime_now_cycles() - reduce_started);
#endif
    }

    float value(int row) const {
        return static_cast<const float*>(
            results.data())[row];
    }
};

struct Q8HotBlockWriter {
    Q4Q8TileDot* dot = nullptr;
    int tile = 0;
    float* values = nullptr;
    bool skip_zero_blocks = false;
    int32_t next_block = 0;
    int32_t fill = 0;

    Q8HotBlockWriter() = default;

    Q8HotBlockWriter(
        Q4Q8TileDot& destination,
        int destination_tile,
        float* hot_values,
        bool skip_zero
    ) {
        reset(
            destination,
            destination_tile,
            hot_values,
            skip_zero);
    }

    __attribute__((always_inline)) inline void reset(
        Q4Q8TileDot& destination,
        int destination_tile,
        float* hot_values,
        bool skip_zero
    ) {
        dot = &destination;
        tile = destination_tile;
        values = hot_values;
        skip_zero_blocks = skip_zero;
        next_block = 0;
        fill = 0;
    }

    __attribute__((always_inline)) inline void flush() {
        dot->quantize_blocks(
            tile,
            next_block,
            values,
            1,
            skip_zero_blocks);
        ++next_block;
        fill = 0;
    }

    __attribute__((always_inline)) inline void append_zeros(
        int32_t count
    ) {
        if (fill != 0) {
            const int32_t take = std::min<int32_t>(
                static_cast<int32_t>(kQ4BlockElements) - fill,
                count);
            std::fill(
                values + fill,
                values + fill + take,
                0.0f);
            fill += take;
            count -= take;
            if (fill == kQ4BlockElements) {
                flush();
            }
        }
        if (count >= kQ4BlockElements) {
            const int32_t complete_blocks =
                count / kQ4BlockElements;
            dot->store_zero_blocks(
                tile, next_block, complete_blocks);
            next_block += complete_blocks;
            count -= complete_blocks * kQ4BlockElements;
        }
        if (count > 0) {
            std::fill(
                values,
                values + count,
                0.0f);
            fill = count;
        }
    }

    __attribute__((always_inline)) inline void append_strided(
        const char* source,
        size_t stride,
        int32_t count
    ) {
        while (count > 0) {
            const int32_t take = std::min<int32_t>(
                static_cast<int32_t>(kQ4BlockElements) - fill,
                count);
            int32_t index = 0;
            for (; index + 4 <= take; index += 4) {
                values[fill + index] =
                    *reinterpret_cast<const float*>(
                        source +
                        static_cast<size_t>(index) * stride);
                values[fill + index + 1] =
                    *reinterpret_cast<const float*>(
                        source +
                        static_cast<size_t>(index + 1) * stride);
                values[fill + index + 2] =
                    *reinterpret_cast<const float*>(
                        source +
                        static_cast<size_t>(index + 2) * stride);
                values[fill + index + 3] =
                    *reinterpret_cast<const float*>(
                        source +
                        static_cast<size_t>(index + 3) * stride);
            }
            for (; index < take; ++index) {
                values[fill + index] =
                    *reinterpret_cast<const float*>(
                        source +
                        static_cast<size_t>(index) *
                            stride);
            }
            source += static_cast<size_t>(take) * stride;
            fill += take;
            count -= take;
            if (fill == kQ4BlockElements) {
                flush();
            }
        }
    }

    __attribute__((always_inline)) inline void append_contiguous(
        const float* source,
        int32_t count
    ) {
        while (count > 0) {
            const int32_t take = std::min<int32_t>(
                static_cast<int32_t>(kQ4BlockElements) - fill,
                count);
            int32_t index = 0;
            for (; index + 4 <= take; index += 4) {
                values[fill + index] = source[index];
                values[fill + index + 1] = source[index + 1];
                values[fill + index + 2] = source[index + 2];
                values[fill + index + 3] = source[index + 3];
            }
            for (; index < take; ++index) {
                values[fill + index] = source[index];
            }
            source += take;
            fill += take;
            count -= take;
            if (fill == kQ4BlockElements) {
                flush();
            }
        }
    }

    __attribute__((always_inline)) inline int32_t block_position()
        const {
        return fill == 0 ? next_block : -1;
    }

    __attribute__((always_inline)) inline bool append_cached_blocks(
        int source_tile,
        int32_t source_first_block,
        int32_t block_count
#if INFLECT_PROFILE_VOCODER_OPS
        ,
        uint64_t source_zero_blocks,
        uint64_t source_zero_values
#endif
    ) {
        if (fill != 0 || block_count < 0) {
            return false;
        }
        dot->copy_blocks(
            tile,
            next_block,
            source_tile,
            source_first_block,
            block_count
#if INFLECT_PROFILE_VOCODER_OPS
            ,
            source_zero_blocks,
            source_zero_values
#endif
        );
        next_block += block_count;
        return true;
    }

    __attribute__((always_inline)) inline bool complete() const {
        return dot != nullptr &&
               fill == 0 &&
               next_block == dot->blocks;
    }
};

struct QuantizedSourceRef {
#if INFLECT_PROFILE_VOCODER_OPS
    uint64_t zero_blocks = 0;
    uint64_t zero_values = 0;
#endif
    int32_t first_block = 0;
    int16_t tile = -1;
    uint16_t reserved = 0;
};

struct PackedQuantDot {
    const ggml_type_traits_cpu* weight_traits = nullptr;
    const ggml_type_traits_cpu* input_traits = nullptr;
    int64_t elements = 0;
    size_t input_bytes = 0;
};

static bool prepare_packed_quant_dot(
    const ggml_tensor* weight,
    int64_t required_elements,
    PackedQuantDot& packed
) {
    if (weight == nullptr || required_elements <= 0 ||
        weight->ne[0] < required_elements) {
        return false;
    }
    packed.weight_traits =
        ggml_get_type_traits_cpu(weight->type);
    if (packed.weight_traits == nullptr ||
        packed.weight_traits->vec_dot == nullptr ||
        packed.weight_traits->vec_dot_type != GGML_TYPE_Q8_0) {
        return false;
    }
    packed.input_traits = ggml_get_type_traits_cpu(
        packed.weight_traits->vec_dot_type);
    if (packed.input_traits == nullptr ||
        packed.input_traits->from_float == nullptr) {
        return false;
    }
    const int64_t block_size = ggml_blck_size(
        packed.weight_traits->vec_dot_type);
    if (block_size <= 0 || weight->ne[0] % block_size != 0) {
        return false;
    }
    packed.elements = weight->ne[0];
    packed.input_bytes = ggml_row_size(
        packed.weight_traits->vec_dot_type,
        packed.elements);
    return packed.input_bytes > 0;
}

static bool quant_conv1d_packed_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    const VocoderQuantConv1dOpData* p,
    const ggml_tensor* x,
    const ggml_tensor* weight,
    const ggml_tensor* bias
) {
    const int64_t T = dst->ne[0];
    const int64_t out_ch = dst->ne[1];
    const int64_t batch = dst->ne[2];
    const int64_t in_ch = x->ne[1];
    const int64_t flat =
        static_cast<int64_t>(p->kernel_size) * in_ch;
    PackedQuantDot packed;
    if (!prepare_packed_quant_dot(weight, flat, packed)) {
        return false;
    }

    VocoderInternalFloatScratch input_window;
    VocoderInternalFloatScratch bias_values;
    VocoderInternalByteScratch hot_input_block;
    VocoderInternalByteScratch channel_span_scratch;
    VocoderInternalByteScratch quantized_windows;
    VocoderInternalByteScratch weight_row;
    std::unique_ptr<Q4Q8TileDot> q4_dot;
    std::unique_ptr<Q8HotBlockWriter[]> tile_writers;
    const int time_tile = runtime_packed_quant_time_tile();
    const bool use_q4_dot =
        weight->type == GGML_TYPE_Q4_0 &&
        runtime_has_s8_dot_blocks_32();
    const size_t weight_row_bytes =
        ggml_row_size(weight->type, packed.elements);
    if (use_q4_dot) {
        q4_dot.reset(new (std::nothrow) Q4Q8TileDot());
        tile_writers.reset(
            new (std::nothrow) Q8HotBlockWriter[time_tile]);
    } else {
        input_window.resize(static_cast<size_t>(packed.elements));
    }
    const int64_t maximum_channel_span =
        static_cast<int64_t>(time_tile - 1) * p->stride +
        static_cast<int64_t>(p->kernel_size - 1) *
            p->dilation +
        1;
    if (use_q4_dot
            ? (q4_dot == nullptr ||
               tile_writers == nullptr ||
               !q4_dot->init(
                   packed.elements, time_tile) ||
               !hot_input_block.try_resize(
                   static_cast<size_t>(time_tile) *
                   kQ4BlockElements * sizeof(float)) ||
               !channel_span_scratch.try_resize(
                   static_cast<size_t>(maximum_channel_span) *
                   sizeof(float)))
            : (!quantized_windows.try_resize(
                   packed.input_bytes *
                       static_cast<size_t>(time_tile)) ||
               !weight_row.try_resize(
                   weight_row_bytes))) {
        return false;
    }

    // The Q4 path spends substantially more time gathering and quantizing
    // input windows than it does in the packed dot product. Split time
    // across workers so each window is quantized once globally. Other
    // quantized formats retain the output-channel split used by ggml.
    const int64_t time_start =
        use_q4_dot ? (T * ith) / nth : 0;
    const int64_t time_end =
        use_q4_dot ? (T * (ith + 1)) / nth : T;
    const int64_t out_start =
        use_q4_dot ? 0 : (out_ch * ith) / nth;
    const int64_t out_end =
        use_q4_dot ? out_ch : (out_ch * (ith + 1)) / nth;
    if (bias != nullptr) {
        bias_values.resize(
            static_cast<size_t>(out_end - out_start));
        for (int64_t o = out_start; o < out_end; ++o) {
            bias_values.data()[o - out_start] =
                tensor_get_f32(bias, o, 0, 0);
        }
    }

    const char* x_data = static_cast<const char*>(x->data);
    char* dst_data = static_cast<char*>(dst->data);
    float* window = use_q4_dot
        ? static_cast<float*>(hot_input_block.data())
        : input_window.data();
#if INFLECT_PROFILE_VOCODER_OPS
    uint64_t output_write_cycles = 0;
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t tile_start = time_start;
             tile_start < time_end;
             tile_start += time_tile) {
            const int tile_count = static_cast<int>(
                std::min<int64_t>(
                    time_tile, time_end - tile_start));
            if (use_q4_dot) {
#if INFLECT_PROFILE_VOCODER_OPS
                const uint64_t quant_cycles_before =
                    q4_dot->input_quant_cycles;
                const uint32_t gather_started =
                    runtime_now_cycles();
#endif
                // Neighboring Conv1D windows overlap heavily. Stage each
                // channel's full temporal span once, then assemble every
                // window in the tile from internal memory. This changes the
                // expensive source reads from tile_count * kernel_size to
                // only the union of their source positions.
                const int64_t first_source_t =
                    tile_start * p->stride - p->padding;
                const int64_t channel_span_count =
                    static_cast<int64_t>(tile_count - 1) *
                        p->stride +
                    static_cast<int64_t>(p->kernel_size - 1) *
                        p->dilation +
                    1;
                const bool stage_channel_span =
                    channel_span_count <
                    static_cast<int64_t>(tile_count) *
                        p->kernel_size;
                if (stage_channel_span) {
                    for (int tile = 0;
                         tile < tile_count;
                         ++tile) {
                        tile_writers[tile].reset(
                            *q4_dot,
                            tile,
                            window +
                                static_cast<size_t>(tile) *
                                    kQ4BlockElements,
                            false);
                    }
                    auto* channel_span =
                        static_cast<float*>(
                            channel_span_scratch.data());
                    const int64_t valid_span_begin =
                        std::min<int64_t>(
                            channel_span_count,
                            std::max<int64_t>(
                                0, -first_source_t));
                    const int64_t valid_span_end =
                        std::max(
                            valid_span_begin,
                            std::min<int64_t>(
                                channel_span_count,
                                x->ne[0] - first_source_t));
                    std::fill(
                        channel_span,
                        channel_span + valid_span_begin,
                        0.0f);
                    std::fill(
                        channel_span + valid_span_end,
                        channel_span + channel_span_count,
                        0.0f);
                    const int64_t valid_count =
                        valid_span_end - valid_span_begin;
                    const char* channel_source =
                        valid_count > 0
                            ? x_data +
                                  (first_source_t +
                                   valid_span_begin) *
                                      x->nb[0] +
                                  b * x->nb[2]
                            : nullptr;
                    for (int32_t channel = 0;
                         channel < in_ch;
                         ++channel) {
                        if (valid_count > 0) {
                            if (x->nb[0] == sizeof(float)) {
                                std::memcpy(
                                    channel_span +
                                        valid_span_begin,
                                    channel_source,
                                    static_cast<size_t>(
                                        valid_count) *
                                        sizeof(float));
                            } else {
                                for (int64_t index = 0;
                                     index < valid_count;
                                     ++index) {
                                    channel_span[
                                        valid_span_begin +
                                        index] =
                                        *reinterpret_cast<
                                            const float*>(
                                                channel_source +
                                                static_cast<size_t>(
                                                    index) *
                                                    x->nb[0]);
                                }
                            }
                            channel_source += x->nb[1];
                        }
                        for (int tile = 0;
                             tile < tile_count;
                             ++tile) {
                            const float* tile_source =
                                channel_span +
                                static_cast<int64_t>(tile) *
                                    p->stride;
                            if (p->dilation == 1) {
                                tile_writers[tile].
                                    append_contiguous(
                                        tile_source,
                                        p->kernel_size);
                            } else {
                                tile_writers[tile].
                                    append_strided(
                                        reinterpret_cast<
                                            const char*>(
                                                tile_source),
                                        static_cast<size_t>(
                                            p->dilation) *
                                            sizeof(float),
                                        p->kernel_size);
                            }
                        }
                    }
                    for (int tile = 0;
                         tile < tile_count;
                         ++tile) {
                        tile_writers[tile].append_zeros(
                            static_cast<int32_t>(
                                packed.elements - flat));
                        if (!tile_writers[tile].complete()) {
                            return false;
                        }
                    }
                } else {
                    for (int tile = 0;
                         tile < tile_count;
                         ++tile) {
                        const int64_t tile_first_source_t =
                            (tile_start + tile) * p->stride -
                            p->padding;
                        const int valid_begin =
                            tile_first_source_t >= 0
                                ? 0
                                : static_cast<int>(
                                      (-tile_first_source_t +
                                       p->dilation - 1) /
                                      p->dilation);
                        const int valid_end =
                            tile_first_source_t >= x->ne[0]
                                ? 0
                                : static_cast<int>(
                                      (x->ne[0] - 1 -
                                       tile_first_source_t) /
                                          p->dilation +
                                      1);
                        const int begin = std::max(
                            0,
                            std::min(
                                p->kernel_size,
                                valid_begin));
                        const int end = std::max(
                            begin,
                            std::min(
                                p->kernel_size,
                                valid_end));
                        Q8HotBlockWriter writer{
                            *q4_dot, tile, window, false};
                        const int64_t source_t =
                            tile_first_source_t +
                            static_cast<int64_t>(begin) *
                                p->dilation;
                        const char* channel_source =
                            end > begin
                                ? x_data +
                                      source_t * x->nb[0] +
                                      b * x->nb[2]
                                : nullptr;
                        for (int32_t channel = 0;
                             channel < in_ch;
                             ++channel) {
                            writer.append_zeros(begin);
                            if (end > begin) {
                                if (p->dilation == 1 &&
                                    x->nb[0] ==
                                        sizeof(float)) {
                                    writer.append_contiguous(
                                        reinterpret_cast<
                                            const float*>(
                                                channel_source),
                                        end - begin);
                                } else {
                                    writer.append_strided(
                                        channel_source,
                                        static_cast<size_t>(
                                            p->dilation) *
                                            x->nb[0],
                                        end - begin);
                                }
                                channel_source += x->nb[1];
                            }
                            writer.append_zeros(
                                p->kernel_size - end);
                        }
                        writer.append_zeros(
                            static_cast<int32_t>(
                                packed.elements - flat));
                        if (!writer.complete()) {
                            return false;
                        }
                    }
                }
#if INFLECT_PROFILE_VOCODER_OPS
                const uint32_t gather_finished =
                    runtime_now_cycles();
                const uint64_t quant_cycles =
                    q4_dot->input_quant_cycles -
                    quant_cycles_before;
                const uint64_t window_cycles =
                    static_cast<uint32_t>(
                        gather_finished - gather_started);
                q4_dot->input_gather_cycles +=
                    window_cycles > quant_cycles
                        ? window_cycles - quant_cycles
                        : 0;
#endif
            } else {
                for (int tile = 0; tile < tile_count; ++tile) {
                    const int64_t t = tile_start + tile;
                    for (int64_t c = 0; c < in_ch; ++c) {
                        float* dst_c =
                            window + c * p->kernel_size;
                        for (int64_t k = 0;
                             k < p->kernel_size;
                             ++k) {
                            const int64_t src_t =
                                t * p->stride +
                                k * p->dilation -
                                p->padding;
                            if (src_t < 0 ||
                                src_t >= x->ne[0]) {
                                dst_c[k] = 0.0f;
                            } else {
                                const char* src =
                                    x_data +
                                    src_t * x->nb[0] +
                                    c * x->nb[1] +
                                    b * x->nb[2];
                                dst_c[k] =
                                    *reinterpret_cast<
                                        const float*>(src);
                            }
                        }
                    }
                    std::fill(
                        window + flat,
                        window + packed.elements,
                        0.0f);
                    void* quantized =
                        static_cast<char*>(
                            quantized_windows.data()) +
                        static_cast<size_t>(tile) *
                            packed.input_bytes;
                    packed.input_traits->from_float(
                        window,
                        quantized,
                        packed.elements);
                }
            }

            for (int64_t o = out_start; o < out_end; ++o) {
                const char* source_weight =
                    static_cast<const char*>(weight->data) +
                    o * weight->nb[1];
                if (use_q4_dot) {
                    q4_dot->unpack_weight(source_weight);
                    q4_dot->calculate(tile_count);
                } else {
                    std::memcpy(
                        weight_row.data(),
                        source_weight,
                        weight_row_bytes);
                }
                for (int tile = 0;
                     tile < tile_count;
                     ++tile) {
                    const int64_t t = tile_start + tile;
                    float value;
                    if (use_q4_dot) {
                        value = q4_dot->value(tile);
                    } else {
                        const void* quantized =
                            static_cast<const char*>(
                                quantized_windows.data()) +
                            static_cast<size_t>(tile) *
                                packed.input_bytes;
                        value = 0.0f;
                        packed.weight_traits->vec_dot(
                            static_cast<int>(
                                packed.elements),
                            &value,
                            0,
                            weight_row.data(),
                            0,
                            quantized,
                            0,
                            1);
                    }
                    if (bias != nullptr) {
                        value +=
                            bias_values.data()[
                                o - out_start];
                    }
                    float* output = reinterpret_cast<float*>(
                        dst_data +
                        t * dst->nb[0] +
                        o * dst->nb[1] +
                        b * dst->nb[2]);
#if INFLECT_PROFILE_VOCODER_OPS
                    if (use_q4_dot) {
                        const uint32_t write_started =
                            runtime_now_cycles();
                        *output = value;
                        output_write_cycles +=
                            static_cast<uint32_t>(
                                runtime_now_cycles() -
                                write_started);
                    } else {
                        *output = value;
                    }
#else
                    *output = value;
#endif
                }
            }
            runtime_cooperate();
        }
    }
#if INFLECT_PROFILE_VOCODER_OPS
    if (use_q4_dot) {
        vocode_profile_add_packed(
            q4_dot->input_gather_cycles,
            q4_dot->reused_input_blocks,
            q4_dot->input_quant_cycles,
            q4_dot->input_max_cycles,
            q4_dot->input_scale_cycles,
            q4_dot->input_convert_cycles,
            q4_dot->q4_unpack_cycles,
            q4_dot->s8_dot_cycles,
            q4_dot->scale_reduce_cycles,
            output_write_cycles);
        vocode_profile_add_input_sparsity(
            p->profile_label,
            q4_dot->profiled_input_blocks,
            q4_dot->profiled_zero_input_blocks,
            q4_dot->profiled_input_values,
            q4_dot->profiled_zero_input_values);
    }
#endif
    return true;
}
#endif

static bool quant_conv1d_resblock_im2col_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    const VocoderQuantConv1dOpData* p,
    const ggml_tensor* x,
    const ggml_tensor* weight,
    const ggml_tensor* bias,
    const ggml_type_traits* traits
) {
    const int64_t T = dst->ne[0];
    const int64_t out_ch = dst->ne[1];
    const int64_t batch = dst->ne[2];
    const int64_t in_ch = x->ne[1];
    const int64_t flat = (int64_t)p->kernel_size * in_ch;
#if defined(INFLECT_LOW_MEMORY)
    const int64_t out_start = (out_ch * ith) / nth;
    const int64_t out_end = (out_ch * (ith + 1)) / nth;
#else
    const int64_t out_start = 0;
    const int64_t out_end = out_ch;
#endif

    VocoderFloatScratch weight_rows;
    VocoderInternalFloatScratch input_window;
    VocoderFloatScratch bias_values;
    const size_t weight_count =
        static_cast<size_t>(weight->ne[0]) *
        static_cast<size_t>(out_end - out_start);
#if defined(INFLECT_LOW_MEMORY)
    if (!weight_rows.try_resize(weight_count)) {
        return false;
    }
#else
    weight_rows.resize(weight_count);
#endif
    input_window.resize((size_t)flat);
    if (bias != nullptr) {
        bias_values.resize((size_t)(out_end - out_start));
    }

    float* weights_f = weight_rows.data();
    for (int64_t o = out_start; o < out_end; o++) {
        const int64_t local_o = o - out_start;
        const char* row_ptr = static_cast<const char*>(weight->data) + o * weight->nb[1];
        traits->to_float(
            row_ptr,
            weights_f + local_o * weight->ne[0],
            weight->ne[0]);
        if (bias != nullptr) {
            bias_values.data()[local_o] =
                tensor_get_f32(bias, o, 0, 0);
        }
    }

    const char* x_data = static_cast<const char*>(x->data);
    char* dst_data = static_cast<char*>(dst->data);
#if defined(INFLECT_LOW_MEMORY)
    const int64_t t_start = 0;
    const int64_t t_end = T;
#else
    const int64_t t_start = (T * ith) / nth;
    const int64_t t_end = (T * (ith + 1)) / nth;
#endif

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t t = t_start; t < t_end; t++) {
            float* window = input_window.data();
            for (int64_t c = 0; c < in_ch; c++) {
                float* dst_c = window + c * p->kernel_size;
                for (int64_t k = 0; k < p->kernel_size; k++) {
                    const int64_t src_t = t * p->stride + k * p->dilation - p->padding;
                    if (src_t < 0 || src_t >= x->ne[0]) {
                        dst_c[k] = 0.0f;
                    } else {
                        const char* src = x_data + src_t * x->nb[0] + c * x->nb[1] + b * x->nb[2];
                        dst_c[k] = *reinterpret_cast<const float*>(src);
                    }
                }
            }

            for (int64_t o = out_start; o < out_end; o++) {
                const int64_t local_o = o - out_start;
                const float* w_row =
                    weights_f + local_o * weight->ne[0];
                const float v =
                    (bias != nullptr
                         ? bias_values.data()[local_o]
                         : 0.0f) +
                    dot_f32_strided(
                        window, 1, w_row, 1, (int)flat);
                *reinterpret_cast<float*>(dst_data + t * dst->nb[0] + o * dst->nb[1] + b * dst->nb[2]) = v;
            }
#if defined(INFLECT_LOW_MEMORY)
            runtime_cooperate();
#endif
        }
    }
    return true;
}

static void quant_conv1d_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata
) {
    const uint32_t op_start_ms = now_ms();
    const auto* p = static_cast<const VocoderQuantConv1dOpData*>(userdata);
    const ggml_tensor* x = dst->src[0];      // [T, in_ch, B]
    const ggml_tensor* weight = dst->src[1]; // [K*in_ch padded, out_ch]
    const ggml_tensor* bias = dst->src[2];
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_quantized(weight->type)) {
        fprintf(stderr, "[VocoderModel] unsupported low-memory conv1d tensor types\n");
        std::abort();
    }

    const int64_t T = dst->ne[0];
    const int64_t out_ch = dst->ne[1];
    const int64_t batch = dst->ne[2];
    const int64_t in_ch = x->ne[1];

    const auto* traits = ggml_get_type_traits(weight->type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "[VocoderModel] unsupported quantized conv1d weight type %s\n",
                ggml_type_name(weight->type));
        std::abort();
    }

#if defined(INFLECT_LOW_MEMORY)
    if (quant_conv1d_packed_op(
            dst, ith, nth, p, x, weight, bias)) {
        const uint64_t out_start =
            (static_cast<uint64_t>(out_ch) * ith) / nth;
        const uint64_t out_end =
            (static_cast<uint64_t>(out_ch) * (ith + 1)) / nth;
        const uint64_t outputs =
            static_cast<uint64_t>(T) *
            (out_end - out_start) *
            static_cast<uint64_t>(batch);
        const uint64_t macs =
            outputs *
            static_cast<uint64_t>(p->kernel_size) *
            static_cast<uint64_t>(in_ch);
        vocode_profile_add_conv1d(
            p->profile_label,
            now_ms() - op_start_ms,
            outputs,
            macs);
        return;
    }
#endif

    if (p->stride == 1 && is_resblock_conv_label(p->profile_label)) {
        if (quant_conv1d_resblock_im2col_op(
                dst, ith, nth, p, x, weight, bias, traits)) {
#if defined(INFLECT_LOW_MEMORY)
            const uint64_t out_start =
                (static_cast<uint64_t>(out_ch) * ith) / nth;
            const uint64_t out_end =
                (static_cast<uint64_t>(out_ch) * (ith + 1)) / nth;
            const uint64_t outputs =
                static_cast<uint64_t>(T) *
                (out_end - out_start) *
                static_cast<uint64_t>(batch);
#else
            const uint64_t t_start = (uint64_t)((T * ith) / nth);
            const uint64_t t_end = (uint64_t)((T * (ith + 1)) / nth);
            const uint64_t outputs =
                (t_end - t_start) * (uint64_t)out_ch * (uint64_t)batch;
#endif
            const uint64_t macs =
                outputs * (uint64_t)p->kernel_size * (uint64_t)in_ch;
            vocode_profile_add_conv1d(
                p->profile_label, now_ms() - op_start_ms, outputs, macs);
            return;
        }
    }

    thread_local VocoderFloatScratch weight_row;
    weight_row.resize(weight->ne[0]);

    const char* x_data = static_cast<const char*>(x->data);
    char* dst_data = static_cast<char*>(dst->data);

    const int64_t o_start = (out_ch * ith) / nth;
    const int64_t o_end = (out_ch * (ith + 1)) / nth;
    for (int64_t o = o_start; o < o_end; o++) {
        const char* row_ptr = static_cast<const char*>(weight->data) + o * weight->nb[1];
        traits->to_float(row_ptr, weight_row.data(), weight->ne[0]);
        const float bias_v = bias != nullptr ? tensor_get_f32(bias, o, 0, 0) : 0.0f;

        for (int64_t b = 0; b < batch; b++) {
            for (int64_t t = 0; t < T; t++) {
                float v = bias_v;
                for (int64_t k = 0; k < p->kernel_size; k++) {
                    const int64_t src_t = t * p->stride + k * p->dilation - p->padding;
                    if (src_t < 0 || src_t >= x->ne[0]) {
                        continue;
                    }
                    const char* x_row = x_data + src_t * x->nb[0] + b * x->nb[2];
                    const auto* x_f = reinterpret_cast<const float*>(x_row);
                    v += dot_f32_strided(x_f, (int)(x->nb[1] / sizeof(float)),
                                         weight_row.data() + k, p->kernel_size,
                                         (int)in_ch);
                }
                *reinterpret_cast<float*>(dst_data + t * dst->nb[0] + o * dst->nb[1] + b * dst->nb[2]) = v;
#if defined(INFLECT_LOW_MEMORY)
                runtime_cooperate();
#endif
            }
        }
    }
    const uint64_t outputs = (uint64_t)(o_end - o_start) * (uint64_t)T * (uint64_t)batch;
    const uint64_t macs = outputs * (uint64_t)p->kernel_size * (uint64_t)in_ch;
    vocode_profile_add_conv1d(p->profile_label, now_ms() - op_start_ms, outputs, macs);
}

static ggml_tensor* add_channel_bias(ggml_context* gctx, ggml_tensor* x, ggml_tensor* bias) {
    ggml_tensor* b = channel_param_3d(gctx, bias);
    if (b->type != x->type && !ggml_is_quantized(b->type)) {
        b = ggml_cast(gctx, b, x->type);
    }
    return ggml_add(gctx, x, b);
}

static ggml_tensor* crop_time_3d(ggml_context* gctx, ggml_tensor* x, int left, int right) {
    if (left == 0 && right == 0) {
        return x;
    }
    const int64_t out_t = x->ne[0] - left - right;
    if (out_t <= 0) {
        fprintf(stderr, "[VocoderModel] invalid temporal crop\n");
        std::abort();
    }
    x = ggml_view_3d(gctx, x, out_t, x->ne[1], x->ne[2],
                     x->nb[1], x->nb[2], left * x->nb[0]);
    return ggml_cont(gctx, x);
}

static std::string debug_dump_dir() {
    const char* dir = std::getenv("INFLECT_DUMP_DIR");
    return (dir && dir[0]) ? std::string(dir) : std::string();
}

static void ensure_debug_dir(const std::string& dir) {
    if (dir.empty()) return;
    std::string cur;
    for (char c : dir) {
        cur.push_back(c);
        if (c == '/' && cur.size() > 1) {
            mkdir(cur.c_str(), 0755);
        }
    }
    mkdir(dir.c_str(), 0755);
}

static std::string debug_path(const std::string& dir, const std::string& name) {
    return dir + "/" + name;
}

static void mem_trace_top_graph_tensors(const char* label, ggml_cgraph* graph, int limit = 16) {
    if (!mem_trace_enabled()) return;
    const char* detail = std::getenv("INFLECT_MEM_TRACE_DETAIL");
    if (!detail || detail[0] != '1') return;

    struct Entry {
        size_t bytes;
        ggml_tensor* tensor;
    };
    std::vector<Entry> entries;
    const int n_nodes = ggml_graph_n_nodes(graph);
    entries.reserve(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        entries.push_back({ggml_nbytes(node), node});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.bytes > b.bytes;
    });

    const int n = std::min(limit, (int)entries.size());
    fprintf(stderr, "[mem] %s largest graph tensors:\n", label);
    for (int i = 0; i < n; i++) {
        const ggml_tensor* t = entries[i].tensor;
        const char* name = t->name[0] ? t->name : "(unnamed)";
        fprintf(stderr,
                "[mem]   %2d %8zu bytes %-12s %-24s shape=[%lld,%lld,%lld,%lld]\n",
                i + 1,
                entries[i].bytes,
                ggml_op_name(t->op),
                name,
                (long long)t->ne[0],
                (long long)t->ne[1],
                (long long)t->ne[2],
                (long long)t->ne[3]);
    }
}

static bool reserve_and_alloc_graph(ggml_gallocr_t allocr, ggml_cgraph* graph) {
    if (!ggml_gallocr_reserve(allocr, graph) || !ggml_gallocr_alloc_graph(allocr, graph)) {
        fprintf(stderr, "[VocoderModel] Failed to allocate vocoder graph\n");
        return false;
    }
    return true;
}

static void debug_manifest(const std::string& dir, const std::string& line) {
    if (dir.empty()) return;
    std::ofstream f(debug_path(dir, "manifest.txt"), std::ios::app);
    f << line << "\n";
}

static void debug_save_f32(const std::string& dir, const std::string& name,
                           const std::vector<float>& data, const std::string& shape) {
    if (dir.empty()) return;
    std::ofstream f(debug_path(dir, name + ".f32"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    debug_manifest(dir, name + " f32 " + shape + " count=" + std::to_string(data.size()));
}

static std::vector<float> debug_transpose_time_channel(const std::vector<float>& data, int T, int C) {
    std::vector<float> out(C * T);
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < T; t++) {
            out[c * T + t] = data[t + c * T];
        }
    }
    return out;
}

static ggml_tensor* conv1d_vocoder(
    ggml_context* gctx,
    ggml_tensor* weight,
    ggml_tensor* bias,
    ggml_tensor* x,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool apply_optional_biases,
    std::vector<VocoderQuantConv1dOpData>& op_data,
    const char* profile_label
) {
    if (!ggml_is_quantized(weight->type)) {
        ggml_tensor* kernel = weight;
        if (kernel->type != GGML_TYPE_F16) {
            kernel = ggml_cast(gctx, kernel, GGML_TYPE_F16);
        }
        ggml_tensor* y = ggml_conv_1d(gctx, kernel, x, stride, padding, dilation);
        if (bias != nullptr) {
            y = add_channel_bias(gctx, y, bias);
        }
        return y;
    }

    const int64_t in_ch = x->ne[1];
    const int64_t out_ch = weight->ne[1];
    const int64_t flat = kernel_size * in_ch;
    if (weight->ne[0] < flat) {
        fprintf(stderr, "[VocoderModel] quantized conv weight too small: weight=[%lld,%lld] flat=%lld\n",
                (long long)weight->ne[0], (long long)weight->ne[1], (long long)flat);
        std::abort();
    }

#if defined(INFLECT_LOW_MEMORY)
    (void)apply_optional_biases;
    const int64_t out_t = (x->ne[0] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    op_data.push_back({profile_label, kernel_size, stride, padding, dilation});
    ggml_tensor* args[] = {x, weight, bias};
    return ggml_custom_4d(gctx, GGML_TYPE_F32, out_t, out_ch, x->ne[2], 1,
                          args, bias != nullptr ? 3 : 2,
                          quant_conv1d_op, GGML_N_TASKS_MAX, &op_data.back());
#else
    // ggml_im2col only needs this tensor for shape/type; weights are consumed
    // by the padded quantized matrix multiply below. Use F32 columns because
    // this GGML CPU PAD op only supports F32.
    ggml_tensor* shape_kernel = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, kernel_size, in_ch, out_ch);
    ggml_tensor* im2col = ggml_im2col(gctx, shape_kernel, x, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor* cols = ggml_reshape_2d(gctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    if (weight->ne[0] > cols->ne[0]) {
        cols = ggml_pad(gctx, cols, weight->ne[0] - cols->ne[0], 0, 0, 0);
    }
    ggml_tensor* y = ggml_mul_mat(gctx, weight, cols);
    y = ggml_cont(gctx, ggml_transpose(gctx, y));
    y = ggml_reshape_3d(gctx, y, im2col->ne[1], out_ch, im2col->ne[2]);
    if (apply_optional_biases && bias != nullptr) {
        y = add_channel_bias(gctx, y, bias);
    }
    return y;
#endif
}

#if defined(INFLECT_LOW_MEMORY)
static bool quant_conv_transpose1d_packed_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    const QuantConvTranspose1dOpData* p,
    const ggml_tensor* x,
    const ggml_tensor* weight
) {
    const int64_t out_t = dst->ne[0];
    const int64_t out_ch = dst->ne[1];
    const int64_t batch = dst->ne[2];
    const int64_t in_t = x->ne[0];
    const int64_t in_ch = x->ne[1];
    const int64_t flat =
        static_cast<int64_t>(p->kernel_size) * in_ch;
    PackedQuantDot packed;
    if (!prepare_packed_quant_dot(weight, flat, packed)) {
        return false;
    }

    VocoderInternalFloatScratch input_window;
    VocoderInternalByteScratch hot_input_block;
    VocoderInternalByteScratch source_cache_scratch;
    VocoderInternalByteScratch quantized_windows;
    VocoderInternalByteScratch weight_row;
    std::unique_ptr<Q4Q8TileDot> q4_dot;
    const int time_tile = runtime_packed_quant_time_tile();
    const bool use_q4_dot =
        weight->type == GGML_TYPE_Q4_0 &&
        runtime_has_s8_dot_blocks_32();
    const bool reuse_aligned_sources =
        use_q4_dot &&
        in_ch % kQ4BlockElements == 0;
    const int source_cache_capacity =
        time_tile + p->kernel_size + 1;
    const size_t weight_row_bytes =
        ggml_row_size(weight->type, packed.elements);
    if (use_q4_dot) {
        q4_dot.reset(new (std::nothrow) Q4Q8TileDot());
    } else {
        input_window.resize(static_cast<size_t>(packed.elements));
    }
    if (use_q4_dot
            ? (q4_dot == nullptr ||
               !q4_dot->init(
                   packed.elements, time_tile, true) ||
               !hot_input_block.try_resize(
                   kQ4BlockElements * sizeof(float)) ||
               (reuse_aligned_sources &&
                !source_cache_scratch.try_resize(
                    static_cast<size_t>(
                        source_cache_capacity) *
                    sizeof(QuantizedSourceRef))))
            : (!quantized_windows.try_resize(
                   packed.input_bytes *
                       static_cast<size_t>(time_tile)) ||
               !weight_row.try_resize(
                   weight_row_bytes))) {
        return false;
    }

    // Match the Conv1d Q4 scheduling: each temporal window is gathered and
    // quantized by only one worker, while every output remains independent.
    const int64_t time_start =
        use_q4_dot ? (out_t * ith) / nth : 0;
    const int64_t time_end =
        use_q4_dot ? (out_t * (ith + 1)) / nth : out_t;
    const int64_t out_start =
        use_q4_dot ? 0 : (out_ch * ith) / nth;
    const int64_t out_end =
        use_q4_dot ? out_ch : (out_ch * (ith + 1)) / nth;
    const char* x_data = static_cast<const char*>(x->data);
    char* dst_data = static_cast<char*>(dst->data);
    float* window = use_q4_dot
        ? static_cast<float*>(hot_input_block.data())
        : input_window.data();
#if INFLECT_PROFILE_VOCODER_OPS
    uint64_t output_write_cycles = 0;
#endif

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t tile_start = time_start;
             tile_start < time_end;
             tile_start += time_tile) {
            const int tile_count = static_cast<int>(
                std::min<int64_t>(
                    time_tile, time_end - tile_start));
            QuantizedSourceRef* source_cache = nullptr;
            int64_t source_cache_first = 0;
            int64_t source_cache_count = 0;
            if (reuse_aligned_sources) {
                const int64_t first_full_t =
                    tile_start + p->crop_left;
                const int64_t last_full_t =
                    tile_start + tile_count - 1 +
                    p->crop_left;
                source_cache_first =
                    first_full_t >= p->kernel_size - 1
                        ? (first_full_t -
                           (p->kernel_size - 1)) /
                              p->stride
                        : 0;
                const int64_t source_cache_last =
                    std::min<int64_t>(
                        in_t - 1,
                        last_full_t / p->stride);
                source_cache_count =
                    std::max<int64_t>(
                        0,
                        source_cache_last -
                            source_cache_first +
                            1);
                if (source_cache_count >
                    source_cache_capacity) {
                    return false;
                }
                source_cache =
                    static_cast<QuantizedSourceRef*>(
                        source_cache_scratch.data());
                std::fill(
                    source_cache,
                    source_cache + source_cache_count,
                    QuantizedSourceRef{});
            }
            for (int tile = 0; tile < tile_count; ++tile) {
                const int64_t t = tile_start + tile;
                const int64_t full_t =
                    t + p->crop_left;
                if (use_q4_dot) {
#if INFLECT_PROFILE_VOCODER_OPS
                    const uint64_t quant_cycles_before =
                        q4_dot->input_quant_cycles;
                    const uint32_t gather_started =
                        runtime_now_cycles();
#endif
                    Q8HotBlockWriter writer{
                        *q4_dot, tile, window, true};
                    // Only one kernel phase can map to an input sample.
                    // Enumerate those taps directly and emit each run of
                    // structurally zero taps in a single writer call.
                    const int first_kernel =
                        static_cast<int>(
                            full_t % p->stride);
                    int next_kernel = 0;
                    int64_t source_t =
                        (full_t - first_kernel) / p->stride;
                    for (int kernel = first_kernel;
                         kernel < p->kernel_size;
                         kernel += p->stride, --source_t) {
                        writer.append_zeros(
                            static_cast<int32_t>(
                                kernel - next_kernel) *
                            static_cast<int32_t>(in_ch));
                        if (source_t >= 0 &&
                            source_t < in_t) {
                            QuantizedSourceRef* cached = nullptr;
                            if (reuse_aligned_sources) {
                                const int64_t cache_index =
                                    source_t -
                                    source_cache_first;
                                if (cache_index < 0 ||
                                    cache_index >=
                                        source_cache_count) {
                                    return false;
                                }
                                cached =
                                    source_cache + cache_index;
                            }
                            if (cached != nullptr &&
                                cached->tile >= 0) {
                                if (!writer.append_cached_blocks(
                                        cached->tile,
                                        cached->first_block,
                                        in_ch /
                                            kQ4BlockElements
#if INFLECT_PROFILE_VOCODER_OPS
                                        ,
                                        cached->zero_blocks,
                                        cached->zero_values
#endif
                                    )) {
                                    return false;
                                }
                            } else {
                                const int64_t first_block =
                                    writer.block_position();
                                if (first_block < 0) {
                                    return false;
                                }
#if INFLECT_PROFILE_VOCODER_OPS
                                const uint64_t zero_blocks_before =
                                    q4_dot->
                                        profiled_zero_input_blocks;
                                const uint64_t zero_values_before =
                                    q4_dot->
                                        profiled_zero_input_values;
#endif
                                writer.append_strided(
                                    x_data +
                                        source_t * x->nb[0] +
                                        b * x->nb[2],
                                    x->nb[1],
                                    in_ch);
                                if (cached != nullptr) {
                                    cached->tile = tile;
                                    cached->first_block =
                                        first_block;
#if INFLECT_PROFILE_VOCODER_OPS
                                    cached->zero_blocks =
                                        q4_dot->
                                            profiled_zero_input_blocks -
                                        zero_blocks_before;
                                    cached->zero_values =
                                        q4_dot->
                                            profiled_zero_input_values -
                                        zero_values_before;
#endif
                                }
                            }
                        } else {
                            writer.append_zeros(in_ch);
                        }
                        next_kernel = kernel + 1;
                    }
                    writer.append_zeros(
                        static_cast<int32_t>(
                            p->kernel_size - next_kernel) *
                        static_cast<int32_t>(in_ch));
                    writer.append_zeros(
                        packed.elements - flat);
                    if (!writer.complete()) {
                        return false;
                    }
#if INFLECT_PROFILE_VOCODER_OPS
                    const uint32_t gather_finished =
                        runtime_now_cycles();
                    const uint64_t quant_cycles =
                        q4_dot->input_quant_cycles -
                        quant_cycles_before;
                    const uint64_t window_cycles =
                        static_cast<uint32_t>(
                            gather_finished - gather_started);
                    q4_dot->input_gather_cycles +=
                        window_cycles > quant_cycles
                            ? window_cycles - quant_cycles
                            : 0;
#endif
                } else {
                    std::fill(
                        window,
                        window + packed.elements,
                        0.0f);
                    for (int64_t k = full_t % p->stride;
                         k < p->kernel_size;
                         k += p->stride) {
                        if (k > full_t) {
                            break;
                        }
                        const int64_t src_t_full =
                            (full_t - k) / p->stride;
                        if (src_t_full < 0 ||
                            src_t_full >= in_t) {
                            continue;
                        }
                        for (int64_t c = 0;
                             c < in_ch;
                             ++c) {
                            const char* src =
                                x_data +
                                src_t_full * x->nb[0] +
                                c * x->nb[1] +
                                b * x->nb[2];
                            window[k * in_ch + c] =
                                *reinterpret_cast<
                                    const float*>(src);
                        }
                    }
                    void* quantized =
                        static_cast<char*>(
                            quantized_windows.data()) +
                        static_cast<size_t>(tile) *
                            packed.input_bytes;
                    packed.input_traits->from_float(
                        window,
                        quantized,
                        packed.elements);
                }
            }

            for (int64_t o = out_start; o < out_end; ++o) {
                const char* source_weight =
                    static_cast<const char*>(weight->data) +
                    o * weight->nb[1];
                if (use_q4_dot) {
                    q4_dot->unpack_weight(source_weight);
                    q4_dot->calculate(tile_count);
                } else {
                    std::memcpy(
                        weight_row.data(),
                        source_weight,
                        weight_row_bytes);
                }
                for (int tile = 0;
                     tile < tile_count;
                     ++tile) {
                    const int64_t t = tile_start + tile;
                    float value;
                    if (use_q4_dot) {
                        value = q4_dot->value(tile);
                    } else {
                        const void* quantized =
                            static_cast<const char*>(
                                quantized_windows.data()) +
                            static_cast<size_t>(tile) *
                                packed.input_bytes;
                        value = 0.0f;
                        packed.weight_traits->vec_dot(
                            static_cast<int>(
                                packed.elements),
                            &value,
                            0,
                            weight_row.data(),
                            0,
                            quantized,
                            0,
                            1);
                    }
                    float* output = reinterpret_cast<float*>(
                        dst_data +
                        t * dst->nb[0] +
                        o * dst->nb[1] +
                        b * dst->nb[2]);
#if INFLECT_PROFILE_VOCODER_OPS
                    if (use_q4_dot) {
                        const uint32_t write_started =
                            runtime_now_cycles();
                        *output = value;
                        output_write_cycles +=
                            static_cast<uint32_t>(
                                runtime_now_cycles() -
                                write_started);
                    } else {
                        *output = value;
                    }
#else
                    *output = value;
#endif
                }
            }
            runtime_cooperate();
        }
    }
#if INFLECT_PROFILE_VOCODER_OPS
    if (use_q4_dot) {
        vocode_profile_add_packed(
            q4_dot->input_gather_cycles,
            q4_dot->reused_input_blocks,
            q4_dot->input_quant_cycles,
            q4_dot->input_max_cycles,
            q4_dot->input_scale_cycles,
            q4_dot->input_convert_cycles,
            q4_dot->q4_unpack_cycles,
            q4_dot->s8_dot_cycles,
            q4_dot->scale_reduce_cycles,
            output_write_cycles);
        vocode_profile_add_input_sparsity(
            p->profile_label,
            q4_dot->profiled_input_blocks,
            q4_dot->profiled_zero_input_blocks,
            q4_dot->profiled_input_values,
            q4_dot->profiled_zero_input_values);
    }
#endif
    return true;
}
#endif

static void quant_conv_transpose1d_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata
) {
    const uint32_t op_start_ms = now_ms();
    const auto* p = static_cast<const QuantConvTranspose1dOpData*>(userdata);
    const ggml_tensor* x = dst->src[0];      // [T_in, in_ch, B]
    const ggml_tensor* weight = dst->src[1]; // [K*in_ch padded, out_ch]
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_quantized(weight->type)) {
        fprintf(stderr, "[VocoderModel] unsupported low-memory conv_transpose1d tensor types\n");
        std::abort();
    }

    const int64_t out_t = dst->ne[0];
    const int64_t out_ch = dst->ne[1];
    const int64_t batch = dst->ne[2];
    const int64_t in_t = x->ne[0];
    const int64_t in_ch = x->ne[1];

#if defined(INFLECT_LOW_MEMORY)
    if (quant_conv_transpose1d_packed_op(
            dst, ith, nth, p, x, weight)) {
        const uint64_t out_start =
            (static_cast<uint64_t>(out_ch) * ith) / nth;
        const uint64_t out_end =
            (static_cast<uint64_t>(out_ch) * (ith + 1)) / nth;
        const uint64_t outputs =
            (out_end - out_start) *
            static_cast<uint64_t>(out_t) *
            static_cast<uint64_t>(batch);
        const uint64_t taps_per_output =
            static_cast<uint64_t>(
                (p->kernel_size + p->stride - 1) /
                p->stride);
        const uint64_t macs =
            outputs *
            taps_per_output *
            static_cast<uint64_t>(in_ch);
        vocode_profile_add_conv_transpose(
            p->profile_label,
            now_ms() - op_start_ms,
            outputs,
            macs);
        return;
    }
#endif

    const auto* traits = ggml_get_type_traits(weight->type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "[VocoderModel] unsupported quantized conv_transpose1d weight type %s\n",
                ggml_type_name(weight->type));
        std::abort();
    }

    thread_local VocoderFloatScratch weight_row;
    weight_row.resize(weight->ne[0]);

    const char* x_data = static_cast<const char*>(x->data);
    char* dst_data = static_cast<char*>(dst->data);

    const int64_t o_start = (out_ch * ith) / nth;
    const int64_t o_end = (out_ch * (ith + 1)) / nth;
    for (int64_t o = o_start; o < o_end; o++) {
        const char* row_ptr = static_cast<const char*>(weight->data) + o * weight->nb[1];
        traits->to_float(row_ptr, weight_row.data(), weight->ne[0]);

        for (int64_t b = 0; b < batch; b++) {
            for (int64_t t = 0; t < out_t; t++) {
                const int64_t full_t = t + p->crop_left;
                float v = 0.0f;
                for (int64_t k = full_t % p->stride; k < p->kernel_size; k += p->stride) {
                    if (k > full_t) {
                        break;
                    }
                    const int64_t src_t = (full_t - k) / p->stride;
                    if (src_t < 0 || src_t >= in_t) {
                        continue;
                    }
                    const char* x_row = x_data + src_t * x->nb[0] + b * x->nb[2];
                    const auto* x_f = reinterpret_cast<const float*>(x_row);
                    v += dot_f32_strided(x_f, (int)(x->nb[1] / sizeof(float)),
                                         weight_row.data() + k * in_ch, 1,
                                         (int)in_ch);
                }
                *reinterpret_cast<float*>(dst_data + t * dst->nb[0] + o * dst->nb[1] + b * dst->nb[2]) = v;
#if defined(INFLECT_LOW_MEMORY)
                runtime_cooperate();
#endif
            }
        }
    }
    const uint64_t outputs = (uint64_t)(o_end - o_start) * (uint64_t)out_t * (uint64_t)batch;
    const uint64_t taps_per_output = (uint64_t)((p->kernel_size + p->stride - 1) / p->stride);
    const uint64_t macs = outputs * taps_per_output * (uint64_t)in_ch;
    vocode_profile_add_conv_transpose(p->profile_label, now_ms() - op_start_ms, outputs, macs);
}

static ggml_tensor* quant_or_f16_conv_transpose_1d(
    ggml_context* ctx,
    ggml_tensor* weight,
    ggml_tensor* x,
    int kernel_size,
    int stride,
    int crop,
    std::vector<QuantConvTranspose1dOpData>& op_data,
    const char* profile_label
) {
    if (!ggml_is_quantized(weight->type)) {
        ggml_tensor* kernel = weight;
        if (kernel->type != GGML_TYPE_F16) {
            kernel = ggml_cast(ctx, kernel, GGML_TYPE_F16);
        }
        ggml_tensor* y = ggml_conv_transpose_1d(ctx, kernel, x, stride, 0, 1);
        return crop_time_3d(ctx, y, crop, crop);
    }

    const int64_t in_ch = x->ne[1];
    const int64_t out_ch = weight->ne[1];
    const int64_t flat = kernel_size * in_ch;
    if (weight->ne[0] < flat) {
        fprintf(stderr, "[VocoderModel] quantized upsample weight too small: weight=[%lld,%lld] flat=%lld\n",
                (long long)weight->ne[0], (long long)weight->ne[1], (long long)flat);
        std::abort();
    }

    const int64_t out_t = (x->ne[0] - 1) * stride + kernel_size - 2 * crop;
    if (out_t <= 0) {
        fprintf(stderr, "[VocoderModel] invalid cropped upsample length\n");
        std::abort();
    }
    op_data.push_back({profile_label, kernel_size, stride, crop});
    ggml_tensor* args[] = {x, weight};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, out_t, out_ch, x->ne[2], 1,
                          args, 2, quant_conv_transpose1d_op, GGML_N_TASKS_MAX, &op_data.back());
}

// ═════════════════════════════════════════════════════════════════════════
// Construction
// ═════════════════════════════════════════════════════════════════════════

VocoderModel::VocoderModel(const VocoderConfig& config) : config_(config) {
    const int n_ups = config.upsample_rates.size();
    const int n_res_per_up = config.resblock_kernel_sizes.size();
    weights_.resblocks.resize(n_ups * n_res_per_up);
    weights_.ups_w.resize(n_ups);
    weights_.ups_b.resize(n_ups);
    weights_.up_acts_alpha.resize(n_ups);
}

VocoderModel::~VocoderModel() {
    // ModelLoader owns the weight context; this class only keeps tensor pointers.
}

int VocoderModel::total_upsample() const {
    int total = 1;
    for (int r : config_.upsample_rates) total *= r;
    return total;
}

// ═════════════════════════════════════════════════════════════════════════
// Weight norm folding
// ═════════════════════════════════════════════════════════════════════════

void VocoderModel::fold_weight_norm(
    ggml_tensor* dst,
    ggml_tensor* weight_v,
    ggml_tensor* weight_g,
    int dim0, int dim1, int dim2,
    bool is_transpose
) {
    // For Conv1d:
    //   weight_v: [K, in, out], weight_g: [1, 1, out]
    //   folded[o] = v[:, :, o] * (g[0,0,o] / ||v[:, :, o]||)
    //
    // For ConvTranspose1d:
    //   weight_v: [K, in, out], weight_g: [1, in, 1]
    //   folded[i] = v[:, i, :] * (g[0,i,0] / ||v[:, i, :]||)

    float* v_data = (float*)weight_v->data;
    float* g_data = (float*)weight_g->data;
    float* d_data = (float*)dst->data;

    if (!is_transpose) {
        // Conv1d: normalize per output channel
        int out_ch = dim2;
        int elements_per_out = dim0 * dim1; // K * in
        for (int o = 0; o < out_ch; o++) {
            float norm = 0.0f;
            for (int e = 0; e < elements_per_out; e++) {
                float val = v_data[o * elements_per_out + e];
                norm += val * val;
            }
            norm = std::sqrt(norm) + 1e-8f;
            float scale = g_data[o] / norm;
            for (int e = 0; e < elements_per_out; e++) {
                d_data[o * elements_per_out + e] =
                    v_data[o * elements_per_out + e] * scale;
            }
        }
    } else {
        // ConvTranspose1d: normalize per input channel
        int in_ch = dim1;
        int out_ch = dim2;
        for (int i = 0; i < in_ch; i++) {
            float norm = 0.0f;
            for (int k = 0; k < dim0; k++) {
                for (int o = 0; o < out_ch; o++) {
                    float val = v_data[k * in_ch * out_ch + i * out_ch + o];
                    norm += val * val;
                }
            }
            norm = std::sqrt(norm) + 1e-8f;
            float scale = g_data[i] / norm;
            for (int k = 0; k < dim0; k++) {
                for (int o = 0; o < out_ch; o++) {
                    int idx = k * in_ch * out_ch + i * out_ch + o;
                    d_data[idx] = v_data[idx] * scale;
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Weight loading
// ═════════════════════════════════════════════════════════════════════════

bool VocoderModel::load(ModelLoader& loader) {
    const int n_ups = config_.upsample_rates.size();
    const int n_res = config_.resblock_kernel_sizes.size();
    bool ok = true;

    auto get = [&](const std::string& name) -> ggml_tensor* {
        if (loader.has_tensor(name)) {
            return loader.get_tensor(name);
        }
        static const std::string prefix = "generator.";
        if (name.rfind(prefix, 0) == 0) {
            std::string stripped = name.substr(prefix.size());
            if (loader.has_tensor(stripped)) {
                return loader.get_tensor(stripped);
            }
        }
        fprintf(stderr, "[VocoderModel] Required tensor not found: %s\n", name.c_str());
        ok = false;
        return nullptr;
    };
    auto maybe = [&](const std::string& name) -> ggml_tensor* {
        if (loader.has_tensor(name)) return loader.get_tensor(name);
        return nullptr;
    };
    const std::string root = config_.tensor_prefix.empty()
                                 ? std::string()
                                 : config_.tensor_prefix + ".";

    // ── Conv pre ────────────────────────────────────────────────────
    // Converter folds weight_norm, so runtime expects plain *.weight tensors.
    {
        weights_.conv_pre_w = get(root + "conv_pre.weight");
        weights_.conv_pre_b = config_.optional_biases
                                  ? maybe(root + "conv_pre.bias")
                                  : get(root + "conv_pre.bias");
    }

    // ── Upsampling layers ───────────────────────────────────────────
    for (int i = 0; i < n_ups; i++) {
        std::string prefix = root + "ups." + std::to_string(i);
        weights_.ups_w[i] = get(prefix + ".weight");
        weights_.ups_b[i] = config_.optional_biases
                                ? maybe(prefix + ".bias")
                                : get(prefix + ".bias");

        if (config_.activation == "snake") {
            std::string act_prefix = root + "up_acts." + std::to_string(i);
            weights_.up_acts_alpha[i] = get(act_prefix + ".log_alpha");
        }
    }

    // ── Residual blocks ─────────────────────────────────────────────
    for (int i = 0; i < n_ups; i++) {
        for (int j = 0; j < n_res; j++) {
            int rb_idx = i * n_res + j;
            auto& rb = weights_.resblocks[rb_idx];

            for (int c = 0; c < 3; c++) {
                // convs1: dilated conv, weight [K, out, out]
                std::string p1 = root + "resblocks." + std::to_string(rb_idx) +
                    ".convs1." + std::to_string(c);
                rb.convs1_w.push_back(get(p1 + ".weight"));
                rb.convs1_b.push_back(config_.optional_biases
                                         ? maybe(p1 + ".bias")
                                         : get(p1 + ".bias"));

                // convs2: dilation=1 conv, weight [K, out, out]
                std::string p2 = root + "resblocks." + std::to_string(rb_idx) +
                    ".convs2." + std::to_string(c);
                rb.convs2_w.push_back(get(p2 + ".weight"));
                rb.convs2_b.push_back(config_.optional_biases
                                         ? maybe(p2 + ".bias")
                                         : get(p2 + ".bias"));

                if (config_.activation == "snake") {
                    rb.acts1_alpha.push_back(get(
                        root + "resblocks." + std::to_string(rb_idx) +
                        ".acts1." + std::to_string(c) + ".log_alpha"));
                    rb.acts2_alpha.push_back(get(
                        root + "resblocks." + std::to_string(rb_idx) +
                        ".acts2." + std::to_string(c) + ".log_alpha"));
                } else {
                    rb.acts1_alpha.push_back(nullptr);
                    rb.acts2_alpha.push_back(nullptr);
                }
            }
        }
    }

    // ── Post layers ─────────────────────────────────────────────────
    {
        weights_.conv_post_w = get(root + "conv_post.weight");
        weights_.conv_post_b = config_.optional_biases
                                   ? maybe(root + "conv_post.bias")
                                   : get(root + "conv_post.bias");
        if (config_.activation == "snake") {
            weights_.post_act_alpha = get(root + "post_act.log_alpha");
        }
    }

    if (!ok) {
        fprintf(stderr, "[VocoderModel] Incomplete vocoder GGUF; regenerate with folded weight tensors.\n");
        return false;
    }

    wctx_ = loader.ctx();
    return true;
}

#if defined(INFLECT_LOW_MEMORY)
bool VocoderModel::load_pre_stage(ModelLoader& loader) {
    const std::string root =
        config_.tensor_prefix.empty()
            ? std::string()
            : config_.tensor_prefix + ".";
    weights_.conv_pre_w =
        loader.get_tensor(root + "conv_pre.weight");
    weights_.conv_pre_b =
        loader.has_tensor(root + "conv_pre.bias")
            ? loader.get_tensor(root + "conv_pre.bias")
            : nullptr;
    wctx_ = loader.ctx();
    return weights_.conv_pre_w != nullptr;
}

bool VocoderModel::load_upsample_stage(
    ModelLoader& loader,
    int stage
) {
    const int n_ups =
        static_cast<int>(config_.upsample_rates.size());
    const int n_res =
        static_cast<int>(config_.resblock_kernel_sizes.size());
    if (stage < 0 || stage >= n_ups) {
        return false;
    }

    const std::string root =
        config_.tensor_prefix.empty()
            ? std::string()
            : config_.tensor_prefix + ".";
    const std::string up =
        root + "ups." + std::to_string(stage);
    weights_.ups_w[stage] =
        loader.get_tensor(up + ".weight");
    weights_.ups_b[stage] =
        loader.has_tensor(up + ".bias")
            ? loader.get_tensor(up + ".bias")
            : nullptr;
    bool ok = weights_.ups_w[stage] != nullptr;

    for (int branch = 0; branch < n_res; ++branch) {
        const int rb_index = stage * n_res + branch;
        ResBlockWeights& rb = weights_.resblocks[rb_index];
        rb = {};
        for (int depth = 0; depth < 3; ++depth) {
            const std::string base =
                root + "resblocks." + std::to_string(rb_index);
            const std::string conv1 =
                base + ".convs1." + std::to_string(depth);
            const std::string conv2 =
                base + ".convs2." + std::to_string(depth);
            ggml_tensor* conv1_weight =
                loader.get_tensor(conv1 + ".weight");
            ggml_tensor* conv2_weight =
                loader.get_tensor(conv2 + ".weight");
            rb.convs1_w.push_back(conv1_weight);
            rb.convs1_b.push_back(
                loader.has_tensor(conv1 + ".bias")
                    ? loader.get_tensor(conv1 + ".bias")
                    : nullptr);
            rb.convs2_w.push_back(conv2_weight);
            rb.convs2_b.push_back(
                loader.has_tensor(conv2 + ".bias")
                    ? loader.get_tensor(conv2 + ".bias")
                    : nullptr);
            rb.acts1_alpha.push_back(nullptr);
            rb.acts2_alpha.push_back(nullptr);
            ok = ok && conv1_weight != nullptr &&
                 conv2_weight != nullptr;
        }
    }
    wctx_ = loader.ctx();
    return ok;
}

bool VocoderModel::load_post_stage(ModelLoader& loader) {
    const std::string root =
        config_.tensor_prefix.empty()
            ? std::string()
            : config_.tensor_prefix + ".";
    weights_.conv_post_w =
        loader.get_tensor(root + "conv_post.weight");
    weights_.conv_post_b =
        loader.has_tensor(root + "conv_post.bias")
            ? loader.get_tensor(root + "conv_post.bias")
            : nullptr;
    wctx_ = loader.ctx();
    return weights_.conv_post_w != nullptr;
}
#endif

// ═════════════════════════════════════════════════════════════════════════
// Snake activation: x + sin²(α·x) / α
// ═════════════════════════════════════════════════════════════════════════

ggml_tensor* VocoderModel::snake(
    ggml_context* gctx,
    ggml_tensor* x,
    ggml_tensor* log_alpha
) {
    // α = exp(log_alpha), clamped to [1e-4, 100]
    ggml_tensor* alpha = ggml_exp(gctx, log_alpha);
    // Clamp: use min/max ops
    alpha = ggml_clamp(gctx, alpha, 1e-4f, 100.0f);
    alpha = channel_param_3d(gctx, alpha);

    // α·x
    ggml_tensor* ax = ggml_mul(gctx, x, alpha);

    // sin²(α·x) = (1 - cos(2α·x)) / 2
    // Using sin² directly: sin(ax)²
    ggml_tensor* sin_ax = ggml_sin(gctx, ax);
    ggml_tensor* sin_sq = ggml_sqr(gctx, sin_ax);

    // sin²(α·x) / α
    ggml_tensor* term = ggml_div(gctx, sin_sq, alpha);

    // x + sin²(α·x) / α
    return ggml_add(gctx, x, term);
}

// ═════════════════════════════════════════════════════════════════════════
// ResBlock1
// ═════════════════════════════════════════════════════════════════════════

ggml_tensor* VocoderModel::build_resblock(
    ggml_context* gctx,
    ggml_tensor* x,   // [T, ch, 1] — GGML conv1d input layout
    const ResBlockWeights& w,
    int kernel_size,
    int max_depth
) {
    // ResBlock1:
    //   for each (c1, c2, a1, a2) in zip(convs1, convs2, acts1, acts2):
    //     y = a1(x) → c1(y) → a2(y) → c2(y) → x = x + y

    const int depth = std::max(1, std::min(3, max_depth));
    for (int i = 0; i < depth; i++) {
        int K = kernel_size;
        int dilation = config_.resblock_dilation_sizes[i % config_.resblock_dilation_sizes.size()][i];
        int pad1 = (K * dilation - dilation) / 2;
        int pad2 = (K - 1) / 2; // dilation=1 for convs2

        // a1(x)
        ggml_tensor* y = config_.activation == "snake"
                             ? snake(gctx, x, w.acts1_alpha[i])
                             : ggml_leaky_relu(gctx, x, 0.1f, false);
        // c1(y) — dilated conv
        y = conv1d_vocoder(gctx, w.convs1_w[i], w.convs1_b[i],
                           y, K, 1, pad1, dilation,
                           config_.optional_biases,
                           quant_conv1d_ops_, "resblock.convs1");

        // a2(y)
        y = config_.activation == "snake"
                ? snake(gctx, y, w.acts2_alpha[i])
                : ggml_leaky_relu(gctx, y, 0.1f, false);
        // c2(y) — dilation=1 conv
        y = conv1d_vocoder(gctx, w.convs2_w[i], w.convs2_b[i],
                           y, K, 1, pad2, 1,
                           config_.optional_biases,
                           quant_conv1d_ops_, "resblock.convs2");

        x = ggml_add(gctx, x, y);
    }
    return x;
}

// ═════════════════════════════════════════════════════════════════════════
// Vocoder graph
// ═════════════════════════════════════════════════════════════════════════

ggml_cgraph* VocoderModel::build_vocoder_graph(
    ggml_context* gctx,
    ggml_tensor* mel  // [n_mels, n_frames, 1]
) {
    const int n_ups = config_.upsample_rates.size();
    const int n_res = config_.resblock_kernel_sizes.size();
    const int active_res = std::min(n_res, 3);
    const int active_depth = 3;
    const int init_ch = config_.upsample_initial_channel;
    (void)init_ch;

    fprintf(stderr,
            "[VocoderModel] resblocks=%d/%d depth=%d/3 "
            "residual_convs=%d/%d im2col=%d"
#if defined(INFLECT_LOW_MEMORY)
            " packed_quant_dot=%d s8_dot=%d"
#endif
            "\n",
            active_res,
            n_res,
            active_depth,
            n_ups * active_res * active_depth * 2,
            n_ups * n_res * 3 * 2,
            1
#if defined(INFLECT_LOW_MEMORY)
            , 1
            , runtime_has_s8_dot_blocks_32() ? 1 : 0
#endif
    );

    const bool capture_debug = !debug_dump_dir().empty();
    std::vector<ggml_tensor*> debug_outputs;
    auto capture = [&](const std::string& name, ggml_tensor* t) {
        if (!capture_debug) return;
        ggml_tensor* out = ggml_cpy(gctx, t, ggml_dup_tensor(gctx, t));
        ggml_set_name(out, name.c_str());
        ggml_set_output(out);
        debug_outputs.push_back(out);
    };

    // ── Conv pre ────────────────────────────────────────────────────
    // GGML conv_1d expects input [T, in_ch, B] and weight [K, in_ch, out_ch]
    // Our mel is [n_mels, n_frames, 1] → need to permute to [n_frames, n_mels, 1]
    ggml_tensor* x = ggml_permute(gctx, mel, 1, 0, 2, 3); // [n_frames, n_mels, 1]
    x = ggml_cont(gctx, x);

    x = conv1d_vocoder(gctx, weights_.conv_pre_w, weights_.conv_pre_b,
                       x, 7, 1, 3, 1,
                       config_.optional_biases,
                       quant_conv1d_ops_, "conv_pre");
    capture("vocoder_conv_pre", x);

    // ── Upsampling + ResBlocks ──────────────────────────────────────
    for (int i = 0; i < n_ups; i++) {
        int rate = config_.upsample_rates[i];
        int K = config_.upsample_kernel_sizes[i];
        int pad = (K - rate) / 2;

        x = config_.activation == "snake"
                ? snake(gctx, x, weights_.up_acts_alpha[i])
                : ggml_leaky_relu(gctx, x, 0.1f, false);

        // Transposed conv: upsampling
        // ggml_conv_transpose_1d(ctx, kernel, input, stride, padding, dilation)
        // This GGML revision only supports p0=0 and d0=1 for conv_transpose_1d.
        x = quant_or_f16_conv_transpose_1d(gctx, weights_.ups_w[i], x, K, rate, pad,
                                           quant_conv_transpose_ops_, "upsample");
        if (config_.optional_biases && weights_.ups_b[i] != nullptr) {
            x = add_channel_bias(gctx, x, weights_.ups_b[i]);
        }
        capture("vocoder_upsample_" + std::to_string(i), x);

        // Residual blocks (sum and average)
        ggml_tensor* xs = nullptr;
        for (int j = 0; j < active_res; j++) {
            int rb_idx = i * n_res + j;
            int rb_kernel = config_.resblock_kernel_sizes[j];
            ggml_tensor* rb_out = build_resblock(gctx, x, weights_.resblocks[rb_idx], rb_kernel, active_depth);
            capture("vocoder_resblock_" + std::to_string(i) + "_" + std::to_string(j), rb_out);
            if (j == 0) {
                xs = rb_out;
            } else {
                xs = ggml_add(gctx, xs, rb_out);
            }
        }
        x = ggml_scale(gctx, xs, 1.0f / active_res);
        capture("vocoder_resblock_avg_" + std::to_string(i), x);
    }

    // ── Post ────────────────────────────────────────────────────────
    x = config_.activation == "snake"
            ? snake(gctx, x, weights_.post_act_alpha)
            : ggml_leaky_relu(gctx, x, 0.01f, false);
    capture("vocoder_post_activation", x);
    x = conv1d_vocoder(gctx, weights_.conv_post_w, weights_.conv_post_b,
                       x, 7, 1, 3, 1,
                       config_.optional_biases,
                       quant_conv1d_ops_, "conv_post");
    capture("vocoder_conv_post", x);
    x = ggml_tanh(gctx, x);
    capture("vocoder_tanh", x);

    // ── Build graph ─────────────────────────────────────────────────
    ggml_cgraph* graph = ggml_new_graph_custom(gctx, 1536, false);
    for (ggml_tensor* out : debug_outputs) {
        ggml_build_forward_expand(graph, out);
    }
    ggml_set_name(x, "audio");
    ggml_set_output(x);
    ggml_build_forward_expand(graph, x);

    return graph;
}

#if defined(INFLECT_LOW_MEMORY)
static std::vector<float> execute_vocoder_stage_graph(
    ggml_context* gctx,
    ggml_cgraph* graph,
    ggml_tensor* input,
    const float* input_data,
    size_t input_bytes,
    std::vector<float>* consumed_input,
    ggml_tensor* output,
    ggml_backend_t backend,
    const char* label,
    bool log_completion
) {
    const uint32_t started_at = now_ms();
    ggml_gallocr_t allocr =
        ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
    if (!reserve_and_alloc_graph(allocr, graph)) {
        ggml_gallocr_free(allocr);
        return {};
    }
    mem_trace_graph(label, gctx, allocr);
    mem_trace_top_graph_tensors(label, graph);
    ggml_backend_tensor_set(
        input, input_data, 0, input_bytes);
    if (consumed_input != nullptr) {
        std::vector<float>().swap(*consumed_input);
        mem_release_to_os();
    }
    const ggml_status status =
        ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(
            stderr,
            "[VocoderModel] staged %s computation failed\n",
            label);
        ggml_gallocr_free(allocr);
        return {};
    }

    std::vector<float> result(
        static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(
        output,
        result.data(),
        0,
        ggml_nbytes(output));
    ggml_gallocr_free(allocr);
    mem_release_to_os();
    if (log_completion) {
        std::fprintf(
            stderr,
            "[VocoderModel] staged %s complete "
            "values=%zu elapsed_ms=%u\n",
            label,
            result.size(),
            static_cast<unsigned>(now_ms() - started_at));
    }
    return result;
}

static ggml_context* new_vocoder_stage_context() {
    struct ggml_init_params params = {
        .mem_size = 96 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    return ggml_init(params);
}

std::vector<float> VocoderModel::run_pre_stage(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    ggml_backend_t backend
) {
    if (weights_.conv_pre_w == nullptr ||
        n_mels <= 0 || n_frames <= 0 ||
        mel.size() !=
            static_cast<size_t>(n_mels) * n_frames) {
        return {};
    }
    ggml_context* gctx = new_vocoder_stage_context();
    if (gctx == nullptr) {
        return {};
    }
    quant_conv1d_ops_.clear();
    quant_conv1d_ops_.reserve(1);
    quant_conv_transpose_ops_.clear();

    ggml_tensor* input = ggml_new_tensor_3d(
        gctx, GGML_TYPE_F32, n_mels, n_frames, 1);
    ggml_tensor* x =
        ggml_cont(gctx, ggml_permute(
            gctx, input, 1, 0, 2, 3));
    x = conv1d_vocoder(
        gctx,
        weights_.conv_pre_w,
        weights_.conv_pre_b,
        x,
        7,
        1,
        3,
        1,
        config_.optional_biases,
        quant_conv1d_ops_,
        "conv_pre");
    ggml_set_name(x, "vocoder_staged_pre");
    ggml_set_output(x);
    ggml_cgraph* graph =
        ggml_new_graph_custom(gctx, 128, false);
    ggml_build_forward_expand(graph, x);
    std::vector<float> output =
        execute_vocoder_stage_graph(
            gctx,
            graph,
            input,
            mel.data(),
            mel.size() * sizeof(float),
            nullptr,
            x,
            backend,
            "pre",
            true);
    ggml_free(gctx);
    mem_release_to_os();
    return output;
}

std::vector<float> VocoderModel::run_upsample_stage(
    std::vector<float>& input_data,
    int input_frames,
    int stage,
    ggml_backend_t backend
) {
    const int n_ups =
        static_cast<int>(config_.upsample_rates.size());
    const int n_res =
        static_cast<int>(config_.resblock_kernel_sizes.size());
    if (stage < 0 || stage >= n_ups ||
        input_frames <= 0) {
        return {};
    }

    int full_stage_limit = 48;
    for (int prior = 0; prior < stage; ++prior) {
        full_stage_limit *= config_.upsample_rates[prior];
    }
    if (input_frames <= full_stage_limit) {
        return run_upsample_stage_once(
            input_data,
            input_frames,
            stage,
            backend,
            true);
    }

    static constexpr int core_frames[] = {
        32, 64, 128, 256,
    };
    int output_radius = 0;
    for (int branch = 0; branch < std::min(n_res, 3);
         ++branch) {
        const int kernel =
            config_.resblock_kernel_sizes[branch];
        int branch_radius = 0;
        for (int depth = 0; depth < 3; ++depth) {
            const auto& dilations =
                config_.resblock_dilation_sizes[
                    depth %
                    config_.resblock_dilation_sizes.size()];
            const int dilation = dilations[depth];
            branch_radius +=
                ((kernel - 1) / 2) * (dilation + 1);
        }
        output_radius =
            std::max(output_radius, branch_radius);
    }

    const int rate = config_.upsample_rates[stage];
    const int kernel =
        config_.upsample_kernel_sizes[stage];
    const int input_halo =
        (output_radius + kernel + rate - 1) / rate + 1;
    const int input_channels =
        config_.upsample_initial_channel >> stage;
    const int output_channels =
        config_.upsample_initial_channel >> (stage + 1);
    const int output_frames = input_frames * rate;
    std::vector<float> output(
        static_cast<size_t>(output_frames) *
        output_channels);
    std::vector<float> input_chunk;
    const uint32_t started_at = now_ms();
    int chunk_count = 0;

    for (int core_start = 0; core_start < input_frames;
         core_start += core_frames[stage]) {
        const int core_end =
            std::min(
                input_frames,
                core_start + core_frames[stage]);
        const int input_start =
            std::max(0, core_start - input_halo);
        const int input_end =
            std::min(
                input_frames,
                core_end + input_halo);
        const int chunk_input_frames =
            input_end - input_start;
        input_chunk.resize(
            static_cast<size_t>(chunk_input_frames) *
            input_channels);
        for (int channel = 0;
             channel < input_channels;
             ++channel) {
            std::copy_n(
                input_data.data() +
                    static_cast<size_t>(channel) *
                        input_frames +
                    input_start,
                chunk_input_frames,
                input_chunk.data() +
                    static_cast<size_t>(channel) *
                        chunk_input_frames);
        }

        std::vector<float> chunk_output =
            run_upsample_stage_once(
                input_chunk,
                chunk_input_frames,
                stage,
                backend,
                false);
        const int chunk_output_frames =
            chunk_input_frames * rate;
        const int keep_start =
            (core_start - input_start) * rate;
        const int keep_frames =
            (core_end - core_start) * rate;
        if (chunk_output.size() !=
            static_cast<size_t>(chunk_output_frames) *
                output_channels) {
            return {};
        }
        for (int channel = 0;
             channel < output_channels;
             ++channel) {
            std::copy_n(
                chunk_output.data() +
                    static_cast<size_t>(channel) *
                        chunk_output_frames +
                    keep_start,
                keep_frames,
                output.data() +
                    static_cast<size_t>(channel) *
                        output_frames +
                    core_start * rate);
        }
        ++chunk_count;
    }
    std::vector<float>().swap(input_data);
    mem_release_to_os();
    std::fprintf(
        stderr,
        "[VocoderModel] staged upsample_%d complete "
        "values=%zu chunks=%d elapsed_ms=%u\n",
        stage,
        output.size(),
        chunk_count,
        static_cast<unsigned>(now_ms() - started_at));
    return output;
}

std::vector<float> VocoderModel::run_upsample_stage_once(
    std::vector<float>& input_data,
    int input_frames,
    int stage,
    ggml_backend_t backend,
    bool log_completion
) {
    const int n_ups =
        static_cast<int>(config_.upsample_rates.size());
    const int n_res =
        static_cast<int>(config_.resblock_kernel_sizes.size());
    if (stage < 0 || stage >= n_ups ||
        input_frames <= 0) {
        return {};
    }
    const int input_channels =
        config_.upsample_initial_channel >> stage;
    const int output_channels =
        config_.upsample_initial_channel >> (stage + 1);
    if (input_channels <= 0 || output_channels <= 0 ||
        input_data.size() !=
            static_cast<size_t>(input_frames) *
                input_channels ||
        weights_.ups_w[stage] == nullptr) {
        return {};
    }

    ggml_context* gctx = new_vocoder_stage_context();
    if (gctx == nullptr) {
        return {};
    }
    constexpr int active_depth = 3;
    const int active_res = std::min(n_res, 3);
    quant_conv1d_ops_.clear();
    quant_conv1d_ops_.reserve(
        static_cast<size_t>(active_res) *
        active_depth * 2);
    quant_conv_transpose_ops_.clear();
    quant_conv_transpose_ops_.reserve(1);

    ggml_tensor* input = ggml_new_tensor_3d(
        gctx,
        GGML_TYPE_F32,
        input_frames,
        input_channels,
        1);
    ggml_tensor* x =
        ggml_leaky_relu(gctx, input, 0.1f, true);
    const int rate = config_.upsample_rates[stage];
    const int kernel = config_.upsample_kernel_sizes[stage];
    const int crop = (kernel - rate) / 2;
    x = quant_or_f16_conv_transpose_1d(
        gctx,
        weights_.ups_w[stage],
        x,
        kernel,
        rate,
        crop,
        quant_conv_transpose_ops_,
        "upsample");
    if (config_.optional_biases &&
        weights_.ups_b[stage] != nullptr) {
        x = add_channel_bias(
            gctx, x, weights_.ups_b[stage]);
    }

    ggml_tensor* sum = nullptr;
    for (int branch = 0; branch < active_res; ++branch) {
        const int rb_index = stage * n_res + branch;
        ggml_tensor* branch_output = build_resblock(
            gctx,
            x,
            weights_.resblocks[rb_index],
            config_.resblock_kernel_sizes[branch],
            active_depth);
        sum = sum == nullptr
                  ? branch_output
                  : ggml_add_inplace(
                        gctx, sum, branch_output);
    }
    x = ggml_scale_inplace(
        gctx, sum, 1.0f / active_res);
    ggml_set_name(x, "vocoder_staged_upsample");
    ggml_set_output(x);
    ggml_cgraph* graph =
        ggml_new_graph_custom(gctx, 512, false);
    ggml_build_forward_expand(graph, x);

    const float* source = input_data.data();
    const size_t source_bytes =
        input_data.size() * sizeof(float);
    std::vector<float> output =
        execute_vocoder_stage_graph(
            gctx,
            graph,
            input,
            source,
            source_bytes,
            &input_data,
            x,
            backend,
            ("upsample_" + std::to_string(stage)).c_str(),
            log_completion);
    ggml_free(gctx);
    mem_release_to_os();
    return output;
}

std::vector<float> VocoderModel::run_post_stage(
    std::vector<float>& input_data,
    int n_frames,
    ggml_backend_t backend
) {
    const int n_ups =
        static_cast<int>(config_.upsample_rates.size());
    const int channels =
        config_.upsample_initial_channel >> n_ups;
    if (weights_.conv_post_w == nullptr ||
        n_frames <= 0 || channels <= 0 ||
        input_data.size() !=
            static_cast<size_t>(n_frames) * channels) {
        return {};
    }

    ggml_context* gctx = new_vocoder_stage_context();
    if (gctx == nullptr) {
        return {};
    }
    quant_conv1d_ops_.clear();
    quant_conv1d_ops_.reserve(1);
    quant_conv_transpose_ops_.clear();

    ggml_tensor* input = ggml_new_tensor_3d(
        gctx, GGML_TYPE_F32, n_frames, channels, 1);
    ggml_tensor* x =
        ggml_leaky_relu(gctx, input, 0.01f, true);
    x = conv1d_vocoder(
        gctx,
        weights_.conv_post_w,
        weights_.conv_post_b,
        x,
        7,
        1,
        3,
        1,
        config_.optional_biases,
        quant_conv1d_ops_,
        "conv_post");
    x = ggml_tanh_inplace(gctx, x);
    ggml_set_name(x, "vocoder_staged_audio");
    ggml_set_output(x);
    ggml_cgraph* graph =
        ggml_new_graph_custom(gctx, 128, false);
    ggml_build_forward_expand(graph, x);

    const float* source = input_data.data();
    const size_t source_bytes =
        input_data.size() * sizeof(float);
    std::vector<float> output =
        execute_vocoder_stage_graph(
            gctx,
            graph,
            input,
            source,
            source_bytes,
            &input_data,
            x,
            backend,
            "post",
            true);
    ggml_free(gctx);
    mem_release_to_os();
    return output;
}

bool VocoderModel::vocode_staged(
    const std::string& model_path,
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    ggml_backend_t backend,
    AudioCallback callback
) {
    if (model_path.empty() || !backend || !callback ||
        n_mels != config_.num_mels ||
        n_frames <= 0 ||
        mel.size() !=
            static_cast<size_t>(n_mels) * n_frames) {
        return false;
    }
    const std::string root =
        config_.tensor_prefix.empty()
            ? std::string()
            : config_.tensor_prefix + ".";

    std::unique_ptr<ModelLoader> loader =
        std::make_unique<ModelLoader>();
    if (!loader->load_selected(
            model_path, {root + "conv_pre."}) ||
        !load_pre_stage(*loader)) {
        return false;
    }
    std::vector<float> current =
        run_pre_stage(mel, n_mels, n_frames, backend);
    loader->release_selected();
    mem_release_to_os();
    runtime_trace_heap("v2 staged pre released");
    if (current.empty()) {
        return false;
    }

    int current_frames = n_frames;
    const int n_ups =
        static_cast<int>(config_.upsample_rates.size());
    const int n_res =
        static_cast<int>(config_.resblock_kernel_sizes.size());
    for (int stage = 0; stage < n_ups; ++stage) {
        std::vector<std::string> prefixes = {
            root + "ups." + std::to_string(stage) + ".",
        };
        for (int branch = 0; branch < n_res; ++branch) {
            prefixes.push_back(
                root + "resblocks." +
                std::to_string(stage * n_res + branch) + ".");
        }
        if (!loader->select(prefixes) ||
            !load_upsample_stage(*loader, stage)) {
            return false;
        }
        std::vector<float> next =
            run_upsample_stage(
                current, current_frames, stage, backend);
        loader->release_selected();
        mem_release_to_os();
        runtime_trace_heap(
            ("v2 staged upsample " +
             std::to_string(stage) +
             " released").c_str());
        if (next.empty()) {
            return false;
        }
        current = std::move(next);
        current_frames *= config_.upsample_rates[stage];
    }

    if (!loader->select({root + "conv_post."}) ||
        !load_post_stage(*loader)) {
        return false;
    }
    std::vector<float> audio =
        run_post_stage(current, current_frames, backend);
    loader->release_selected();
    mem_release_to_os();
    runtime_trace_heap("v2 staged post released");
    if (audio.empty()) {
        return false;
    }
    callback(audio.data(), audio.size());
    return true;
}
#endif

// ═════════════════════════════════════════════════════════════════════════
// Full vocoding
// ═════════════════════════════════════════════════════════════════════════

std::vector<float> VocoderModel::vocode(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    ggml_backend_t backend
) {
    const uint32_t start_ms = now_ms();
    uint32_t stage_ms = start_ms;
    fprintf(stderr, "[VocoderModel] vocode begin frames=%d mels=%d samples_est=%d\n",
            n_frames, n_mels, n_frames * total_upsample());

    // Create graph context
#if defined(INFLECT_LOW_MEMORY)
    size_t gctx_size = 480 * 1024;
#else
    size_t gctx_size = 1024 * 1024;
#endif
    struct ggml_init_params gparams = {
        .mem_size   = gctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* gctx = ggml_init(gparams);
    mem_trace_rss("vocoder ctx init");
    mem_trace_heap("vocoder ctx init");

    // Create input tensor
    ggml_tensor* mel_t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_mels, n_frames, 1);

    // Build graph
    quant_conv1d_ops_.clear();
    size_t n_conv1d_ops = 2; // pre + post
    for (const auto& dilations : config_.resblock_dilation_sizes) {
        n_conv1d_ops += config_.upsample_rates.size() * dilations.size() * 2;
    }
    quant_conv1d_ops_.reserve(n_conv1d_ops);
    quant_conv_transpose_ops_.clear();
    quant_conv_transpose_ops_.reserve(config_.upsample_rates.size());
    ggml_cgraph* graph = build_vocoder_graph(gctx, mel_t);
    mem_trace_rss("vocoder graph built");
    mem_trace_heap("vocoder graph built");
    fprintf(stderr, "[VocoderModel] graph built stage_ms=%u total_ms=%u\n",
            (unsigned)(now_ms() - stage_ms),
            (unsigned)(now_ms() - start_ms));
    stage_ms = now_ms();

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!reserve_and_alloc_graph(allocr, graph)) {
        ggml_gallocr_free(allocr);
        ggml_free(gctx);
        return {};
    }
    mem_trace_graph("vocoder", gctx, allocr);
    mem_trace_top_graph_tensors("vocoder", graph);
    mem_trace_rss("vocoder allocated");
    mem_trace_heap("vocoder allocated");
    fprintf(stderr, "[VocoderModel] allocated stage_ms=%u total_ms=%u\n",
            (unsigned)(now_ms() - stage_ms),
            (unsigned)(now_ms() - start_ms));
    stage_ms = now_ms();

    // Set input
    ggml_backend_tensor_set(mel_t, mel.data(), 0, n_mels * n_frames * sizeof(float));
    mem_trace_rss("vocoder input copied");
    mem_trace_heap("vocoder input copied");
    fprintf(stderr, "[VocoderModel] input copied stage_ms=%u total_ms=%u\n",
            (unsigned)(now_ms() - stage_ms),
            (unsigned)(now_ms() - start_ms));
    stage_ms = now_ms();

    // Compute
    vocode_profile_reset();
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    const uint32_t compute_elapsed_ms = now_ms() - stage_ms;
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VocoderModel] Graph computation stopped: %s\n",
                ggml_status_to_string(status));
        ggml_gallocr_free(allocr);
        ggml_free(gctx);
        return {};
    }
    mem_trace_rss("vocoder computed");
    mem_trace_heap("vocoder computed");
    fprintf(stderr, "[VocoderModel] computed stage_ms=%u total_ms=%u\n",
            (unsigned)compute_elapsed_ms,
            (unsigned)(now_ms() - start_ms));
    vocode_profile_log(compute_elapsed_ms);
    stage_ms = now_ms();

    const std::string dump_dir = debug_dump_dir();
    if (!dump_dir.empty()) {
        ensure_debug_dir(dump_dir);
        std::vector<std::string> dump_names = {"vocoder_conv_pre"};
        for (int i = 0; i < (int)config_.upsample_rates.size(); i++) {
            dump_names.push_back("vocoder_upsample_" + std::to_string(i));
            for (int j = 0; j < (int)config_.resblock_kernel_sizes.size(); j++) {
                dump_names.push_back("vocoder_resblock_" + std::to_string(i) + "_" + std::to_string(j));
            }
            dump_names.push_back("vocoder_resblock_avg_" + std::to_string(i));
        }
        dump_names.push_back("vocoder_post_activation");
        dump_names.push_back("vocoder_conv_post");
        dump_names.push_back("vocoder_tanh");

        for (const std::string& name : dump_names) {
            ggml_tensor* t = ggml_get_tensor(gctx, name.c_str());
            if (!t) continue;
            if (t->type != GGML_TYPE_F32 || t->ne[2] != 1) {
                fprintf(stderr, "[VocoderModel] Skipping debug dump for non-F32 tensor %s\n", name.c_str());
                continue;
            }
            const int T = (int)t->ne[0];
            const int C = (int)t->ne[1];
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
            debug_save_f32(
                dump_dir,
                name,
                debug_transpose_time_channel(data, T, C),
                "[" + std::to_string(C) + "," + std::to_string(T) + "]"
            );
        }
    }

    // Extract audio
    ggml_tensor* audio_t = ggml_get_tensor(gctx, "audio");
    if (!audio_t) {
        fprintf(stderr, "[VocoderModel] Failed to locate named vocoder output\n");
        std::abort();
    }
    std::vector<float> audio(ggml_nelements(audio_t));
    ggml_backend_tensor_get(audio_t, audio.data(), 0, ggml_nbytes(audio_t));
    mem_trace_rss("vocoder audio copied");
    mem_trace_heap("vocoder audio copied");
    fprintf(stderr, "[VocoderModel] audio copied samples=%zu stage_ms=%u total_ms=%u\n",
            audio.size(),
            (unsigned)(now_ms() - stage_ms),
            (unsigned)(now_ms() - start_ms));
    stage_ms = now_ms();

    // Cleanup
    ggml_gallocr_free(allocr);
    mem_release_to_os();
    mem_trace_rss("vocoder allocator freed");
    mem_trace_heap("vocoder allocator freed");
    ggml_free(gctx);
    mem_release_to_os();
    mem_trace_rss("vocoder context freed");
    mem_trace_heap("vocoder context freed");
    fprintf(stderr, "[VocoderModel] vocode complete stage_ms=%u total_ms=%u\n",
            (unsigned)(now_ms() - stage_ms),
            (unsigned)(now_ms() - start_ms));

    return audio;
}

// ═════════════════════════════════════════════════════════════════════════
// Chunked vocoding (for MCU: overlap-discard strategy)
// ═════════════════════════════════════════════════════════════════════════

void VocoderModel::vocode_streaming(
    const std::vector<float>& mel,
    int n_mels,
    int n_frames,
    int chunk_frames,
    ggml_backend_t backend,
    AudioCallback callback
) {
    const bool legacy_v1_chunking =
        config_.activation == "snake" &&
        config_.tensor_prefix == "generator";

#if defined(INFLECT_LOW_MEMORY)
    constexpr int kLowMemoryMaxChunkFrames = 96;
    constexpr int kV1LowMemoryMinChunkFrames = 24;
    constexpr int kV2LowMemoryMinChunkFrames = 1;
    constexpr int low_memory_max_chunk_frames =
        kLowMemoryMaxChunkFrames;
    if (chunk_frames <= 0) {
        chunk_frames = low_memory_max_chunk_frames;
    }
    if (chunk_frames > low_memory_max_chunk_frames) {
        chunk_frames = low_memory_max_chunk_frames;
    }
    if (legacy_v1_chunking && n_frames > kLowMemoryMaxChunkFrames) {
        chunk_frames = kLowMemoryMaxChunkFrames;
    } else {
        const int minimum = legacy_v1_chunking
                                ? kV1LowMemoryMinChunkFrames
                                : kV2LowMemoryMinChunkFrames;
        if (chunk_frames > 0 && chunk_frames < minimum) {
            chunk_frames = minimum;
        }
    }
    fprintf(stderr,
            "[VocoderModel] low-memory chunk policy frames=%d "
            "chunk_frames=%d max_chunk_frames=%d\n",
            n_frames, chunk_frames, low_memory_max_chunk_frames);
#endif

#if defined(INFLECT_LOW_MEMORY)
    if (!legacy_v1_chunking &&
        chunk_frames > 0 &&
        chunk_frames < n_frames &&
        n_frames <= kLowMemoryMaxChunkFrames) {
        fprintf(
            stderr,
            "[VocoderModel] trying full short utterance "
            "frames=%d\n",
            n_frames);
        auto audio = vocode(
            mel, n_mels, n_frames, backend);
        if (!audio.empty()) {
            callback(audio.data(), audio.size());
            return;
        }
        if (runtime_cancelled()) {
            return;
        }
        fprintf(
            stderr,
            "[VocoderModel] full short utterance did not fit; "
            "using chunk_frames=%d\n",
            chunk_frames);
    }
#endif

    if (chunk_frames <= 0 || chunk_frames >= n_frames) {
        // No chunking — vocode everything at once
        auto audio = vocode(mel, n_mels, n_frames, backend);
        callback(audio.data(), audio.size());
        return;
    }

    if (legacy_v1_chunking) {
        int overlap = 64;
#if defined(INFLECT_LOW_MEMORY)
        overlap = 4;
#endif
        if (chunk_frames <= overlap) {
            auto audio = vocode(mel, n_mels, n_frames, backend);
            callback(audio.data(), audio.size());
            return;
        }
        const int hop = chunk_frames - overlap;
        const int upsample = total_upsample();
        std::vector<float> mel_chunk;
        mel_chunk.reserve(n_mels * chunk_frames);
        std::vector<float> audio_chunk;
        std::vector<float> pending_tail;
        std::vector<float> blended;

        int chunk_index = 0;
        for (int start = 0; start < n_frames; start += hop) {
            if (runtime_cancelled()) {
                return;
            }
            const uint32_t chunk_start_ms = now_ms();
            const int end = std::min(start + chunk_frames, n_frames);
            const int chunk_len = end - start;
            fprintf(stderr,
                    "[VocoderModel] chunk %d begin mel_start=%d frames=%d\n",
                    chunk_index, start, chunk_len);

            mel_chunk.resize(n_mels * chunk_len);
            for (int m = 0; m < n_mels; m++) {
                for (int t = 0; t < chunk_len; t++) {
                    mel_chunk[m + n_mels * t] =
                        mel[m + n_mels * (start + t)];
                }
            }

            audio_chunk = vocode(
                mel_chunk, n_mels, chunk_len, backend);
            if (runtime_cancelled()) {
                return;
            }
            const int discard_start =
                start > 0 ? overlap * upsample / 2 : 0;
            const int discard_end =
                end < n_frames ? overlap * upsample / 2 : 0;
            const int out_start = discard_start;
            const int out_end =
                static_cast<int>(audio_chunk.size()) - discard_end;

            if (out_end > out_start) {
                const int out_len = out_end - out_start;
                int blended_len = 0;
                if (!pending_tail.empty()) {
                    blended_len = std::min<int>(
                        pending_tail.size(), out_len);
                    blended.resize(blended_len);
                    for (int i = 0; i < blended_len; ++i) {
                        const float amount =
                            static_cast<float>(i + 1) /
                            static_cast<float>(blended_len + 1);
                        blended[i] =
                            pending_tail[i] * (1.0f - amount) +
                            audio_chunk[out_start + i] * amount;
                    }
                    callback(blended.data(), blended.size());
                }
                if (out_len > blended_len) {
                    callback(
                        audio_chunk.data() + out_start + blended_len,
                        out_len - blended_len);
                }
            } else if (!pending_tail.empty() && end >= n_frames) {
                callback(pending_tail.data(), pending_tail.size());
                pending_tail.clear();
            }

            if (discard_end > 0) {
                pending_tail.assign(
                    audio_chunk.begin() + out_end, audio_chunk.end());
            } else {
                pending_tail.clear();
            }
            fprintf(stderr,
                    "[VocoderModel] chunk %d done elapsed_ms=%u\n",
                    chunk_index,
                    static_cast<unsigned>(now_ms() - chunk_start_ms));
            ++chunk_index;
            if (end >= n_frames) break;
        }
        return;
    }

    // Receptive-field-derived latent halo for the v2 HiFi-GAN topology.
    // Back-propagating one output core through conv_post(k=7), the three
    // ResBlock1 pairs at each stage (max k=11, d=1/3/5), transposed
    // convolutions (8/8/2/2), and conv_pre(k=7) reaches at most 14 latent
    // frames beyond either edge. Decode core+halo and emit the core once;
    // no overlap blending or cumulative sample shift is needed.
    constexpr int halo_left_frames = 14;
    constexpr int halo_right_frames = 13;
    int upsample = total_upsample();
    std::vector<float> mel_chunk;
    mel_chunk.reserve(
        static_cast<size_t>(n_mels) *
        (chunk_frames + halo_left_frames + halo_right_frames));
    std::vector<float> audio_chunk;

    int chunk_index = 0;
#if defined(INFLECT_LOW_MEMORY)
    int core_start = 0;
    while (core_start < n_frames) {
#else
    for (int core_start = 0; core_start < n_frames;
         core_start += chunk_frames) {
#endif
        const uint32_t chunk_start_ms = now_ms();
        const int core_end = std::min(core_start + chunk_frames, n_frames);
        const int input_start =
            std::max(0, core_start - halo_left_frames);
        const int input_end =
            std::min(n_frames, core_end + halo_right_frames);
        const int input_frames = input_end - input_start;
        fprintf(stderr,
                "[VocoderModel] chunk %d begin core=[%d,%d) input=[%d,%d)\n",
                chunk_index, core_start, core_end, input_start, input_end);

        mel_chunk.resize(static_cast<size_t>(n_mels) * input_frames);
        for (int m = 0; m < n_mels; m++) {
            for (int t = 0; t < input_frames; t++) {
                mel_chunk[m + n_mels * t] =
                    mel[m + n_mels * (input_start + t)];
            }
        }

        audio_chunk = vocode(
            mel_chunk, n_mels, input_frames, backend);
#if defined(INFLECT_LOW_MEMORY)
        if (audio_chunk.empty()) {
            if (runtime_cancelled()) {
                return;
            }
            if (chunk_frames <= 1) {
                std::fprintf(
                    stderr,
                    "[VocoderModel] V2 decoder allocation failed at "
                    "minimum chunk size\n");
                return;
            }
            chunk_frames = std::max(1, chunk_frames / 2);
            std::fprintf(
                stderr,
                "[VocoderModel] retrying V2 decoder core=%d "
                "chunk_frames=%d\n",
                core_start, chunk_frames);
            continue;
        }
#endif
        const size_t keep_start =
            static_cast<size_t>(core_start - input_start) * upsample;
        const size_t keep_count =
            static_cast<size_t>(core_end - core_start) * upsample;
        if (keep_start + keep_count > audio_chunk.size()) {
            std::fprintf(stderr,
                         "[VocoderModel] haloed chunk output is too short\n");
            return;
        }
        callback(audio_chunk.data() + keep_start, keep_count);
        fprintf(stderr, "[VocoderModel] chunk %d done elapsed_ms=%u\n",
                chunk_index,
                (unsigned)(now_ms() - chunk_start_ms));
        chunk_index++;
#if defined(INFLECT_LOW_MEMORY)
        core_start = core_end;
#endif
    }
}

} // namespace inflect
