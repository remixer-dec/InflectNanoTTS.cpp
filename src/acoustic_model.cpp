#include "acoustic_model.h"
#include "inflect-nano.h"
#include "memory_trace.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <random>
#include <cstdlib>
#include <vector>

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

#ifndef INFLECT_PROFILE_ACOUSTIC_OPS
#define INFLECT_PROFILE_ACOUSTIC_OPS 0
#endif

#ifndef INFLECT_ACOUSTIC_SKIP_POSTNET
#define INFLECT_ACOUSTIC_SKIP_POSTNET 0
#endif

namespace inflect {

struct FloatScratch {
    float* ptr = nullptr;
    size_t cap = 0;

    ~FloatScratch() {
        runtime_free_scratch(ptr);
    }

    void resize(size_t n) {
        if (n <= cap) {
            return;
        }
        float* next = static_cast<float*>(
            runtime_alloc_scratch(n * sizeof(float), ScratchMemoryKind::Psram));
        if (next == nullptr) {
            fprintf(stderr, "[AcousticModel] scratch allocation failed floats=%zu\n", n);
            std::abort();
        }
        runtime_free_scratch(ptr);
        ptr = next;
        cap = n;
    }

    float* data() { return ptr; }
    const float* data() const { return ptr; }
    float& operator[](size_t i) { return ptr[i]; }
    const float& operator[](size_t i) const { return ptr[i]; }
};

#if defined(INFLECT_LOW_MEMORY)
constexpr int kAcousticQ4BlockElements = 32;

struct AcousticPackedQ4Block {
    ggml_fp16_t scale;
    uint8_t quants[kAcousticQ4BlockElements / 2];
};

static_assert(
    sizeof(AcousticPackedQ4Block) ==
        sizeof(ggml_fp16_t) + kAcousticQ4BlockElements / 2,
    "unexpected Q4_0 block layout");

struct AcousticInternalScratch {
    void* allocation = nullptr;
    void* ptr = nullptr;
    size_t cap = 0;

    ~AcousticInternalScratch() {
        runtime_free_scratch(allocation);
    }

    bool try_resize(size_t bytes) {
        if (bytes <= cap) {
            return true;
        }
        runtime_free_scratch(allocation);
        allocation = runtime_alloc_scratch(
            bytes + 15, ScratchMemoryKind::InternalPreferred);
        if (allocation == nullptr) {
            ptr = nullptr;
            cap = 0;
            return false;
        }
        const uintptr_t address =
            reinterpret_cast<uintptr_t>(allocation);
        ptr = reinterpret_cast<void*>(
            (address + 15) & ~uintptr_t(15));
        cap = bytes;
        return true;
    }

    void* data() { return ptr; }
    const void* data() const { return ptr; }
};

#if INFLECT_PROFILE_ACOUSTIC_OPS
struct AcousticPackedPhaseTimes {
    uint64_t total_cycles = 0;
    uint64_t scratch_init_cycles = 0;
    uint64_t cooperate_cycles = 0;
    uint64_t input_quant_cycles = 0;
    uint64_t q4_unpack_cycles = 0;
    uint64_t s8_dot_cycles = 0;
    uint64_t scale_reduce_cycles = 0;
    uint64_t output_write_cycles = 0;
    uint64_t input_rows = 0;
    uint64_t output_rows = 0;
};
#endif

enum class AcousticPackedProfileKind : uint8_t {
    DenseLinear,
    GruInput,
    GruRecurrent,
    BridgeLinear,
    PostnetConv1d,
    Count,
};
#endif

#if INFLECT_PROFILE_ACOUSTIC_OPS
#define ACOUSTIC_PROFILE_START_MS(name) \
    const uint32_t name = runtime_now_ms()

struct AcousticProfileBucket {
    const char* label;
    std::atomic<uint32_t> calls;
    std::atomic<uint64_t> elapsed_ms;
    std::atomic<uint64_t> outputs;
    std::atomic<uint64_t> macs;
};

static AcousticProfileBucket g_acoustic_profile_buckets[] = {
    // Keep reports in optimization priority order.
    {"dense.ff_expand", {}, {}, {}, {}},
    {"dense.ff_project", {}, {}, {}, {}},
    {"dense.point", {}, {}, {}, {}},
    {"dense.mel_hidden", {}, {}, {}, {}},
    {"dense.mel_output", {}, {}, {}, {}},
    {"dense.other", {}, {}, {}, {}},
    {"frame_gru", {}, {}, {}, {}},
    {"depthwise_gated", {}, {}, {}, {}},
    {"layer_norm", {}, {}, {}, {}},
    {"bridge.linear", {}, {}, {}, {}},
    {"postnet.conv1d", {}, {}, {}, {}},
};

struct AcousticPhaseProfileBucket {
    const char* label;
    uint32_t sample_stride;
    std::atomic<uint64_t> cycles;
    std::atomic<uint64_t> units;
};

constexpr uint32_t kAcousticDepthwiseProfileStride = 32;

static AcousticPhaseProfileBucket
    g_acoustic_phase_profile_buckets[] = {
        {"gru.total", 1, {}, {}},
        {"gru.weight_unpack", 1, {}, {}},
        {"gru.state_alloc", 1, {}, {}},
        {"gru.bias_load", 1, {}, {}},
        {"gru.input_projection", 1, {}, {}},
        {"gru.input_dot", 1, {}, {}},
        {"gru.recurrent_projection", 1, {}, {}},
        {"gru.recurrent_dot", 1, {}, {}},
        {"gru.activation", 1, {}, {}},
        {"gru.state_write", 1, {}, {}},
        {"gru.unclassified", 1, {}, {}},
        {"depthwise.prepare", 1, {}, {}},
        {"depthwise.bias_load",
         kAcousticDepthwiseProfileStride, {}, {}},
        {"depthwise.mac",
         kAcousticDepthwiseProfileStride, {}, {}},
        {"depthwise.gate_write",
         kAcousticDepthwiseProfileStride, {}, {}},
        {"layer_norm.total", 1, {}, {}},
        {"layer_norm.mean", 1, {}, {}},
        {"layer_norm.variance", 1, {}, {}},
        {"layer_norm.affine_write", 1, {}, {}},
};

static void acoustic_phase_profile_reset() {
    for (auto& bucket : g_acoustic_phase_profile_buckets) {
        bucket.cycles.store(0);
        bucket.units.store(0);
    }
}

static void acoustic_phase_profile_add(
    const char* label,
    uint64_t cycles,
    uint64_t units
) {
    for (auto& bucket : g_acoustic_phase_profile_buckets) {
        if (std::strcmp(bucket.label, label) == 0) {
            bucket.cycles.fetch_add(cycles);
            bucket.units.fetch_add(units);
            return;
        }
    }
}

static const char* acoustic_dense_profile_label(
    const ggml_tensor* weight
) {
    const char* name =
        weight != nullptr ? weight->name : "";
    if (std::strstr(name, ".ff.0.weight") != nullptr) {
        return "dense.ff_expand";
    }
    if (std::strstr(name, ".ff.3.weight") != nullptr) {
        return "dense.ff_project";
    }
    if (std::strstr(name, ".point.weight") != nullptr) {
        return "dense.point";
    }
    if (std::strcmp(name, "mel_head.1.weight") == 0) {
        return "dense.mel_hidden";
    }
    if (std::strcmp(name, "mel_head.3.weight") == 0) {
        return "dense.mel_output";
    }
    return "dense.other";
}

#if defined(INFLECT_LOW_MEMORY)
struct AcousticPackedProfile {
    const char* label;
    std::atomic<uint64_t> total_cycles{0};
    std::atomic<uint64_t> scratch_init_cycles{0};
    std::atomic<uint64_t> cooperate_cycles{0};
    std::atomic<uint64_t> input_quant_cycles{0};
    std::atomic<uint64_t> q4_unpack_cycles{0};
    std::atomic<uint64_t> s8_dot_cycles{0};
    std::atomic<uint64_t> scale_reduce_cycles{0};
    std::atomic<uint64_t> output_write_cycles{0};
    std::atomic<uint64_t> input_rows{0};
    std::atomic<uint64_t> output_rows{0};
};

static AcousticPackedProfile g_acoustic_packed_profiles[] = {
    {"dense_linear"},
    {"gru_input"},
    {"gru_recurrent"},
    {"bridge_linear"},
    {"postnet_conv1d"},
};

static_assert(
    sizeof(g_acoustic_packed_profiles) /
            sizeof(g_acoustic_packed_profiles[0]) ==
        static_cast<size_t>(AcousticPackedProfileKind::Count),
    "packed acoustic profile labels must match profile kinds");

static void acoustic_packed_profile_reset() {
    for (auto& profile : g_acoustic_packed_profiles) {
        profile.total_cycles.store(0);
        profile.scratch_init_cycles.store(0);
        profile.cooperate_cycles.store(0);
        profile.input_quant_cycles.store(0);
        profile.q4_unpack_cycles.store(0);
        profile.s8_dot_cycles.store(0);
        profile.scale_reduce_cycles.store(0);
        profile.output_write_cycles.store(0);
        profile.input_rows.store(0);
        profile.output_rows.store(0);
    }
}

static void acoustic_packed_profile_add(
    AcousticPackedProfileKind kind,
    const AcousticPackedPhaseTimes& phases
) {
    auto& profile =
        g_acoustic_packed_profiles[static_cast<size_t>(kind)];
    profile.total_cycles.fetch_add(
        phases.total_cycles);
    profile.scratch_init_cycles.fetch_add(
        phases.scratch_init_cycles);
    profile.cooperate_cycles.fetch_add(
        phases.cooperate_cycles);
    profile.input_quant_cycles.fetch_add(
        phases.input_quant_cycles);
    profile.q4_unpack_cycles.fetch_add(
        phases.q4_unpack_cycles);
    profile.s8_dot_cycles.fetch_add(
        phases.s8_dot_cycles);
    profile.scale_reduce_cycles.fetch_add(
        phases.scale_reduce_cycles);
    profile.output_write_cycles.fetch_add(
        phases.output_write_cycles);
    profile.input_rows.fetch_add(
        phases.input_rows);
    profile.output_rows.fetch_add(
        phases.output_rows);
}
#else
static void acoustic_packed_profile_reset() {}
#endif

static void acoustic_profile_reset() {
    for (auto& bucket : g_acoustic_profile_buckets) {
        bucket.calls.store(0);
        bucket.elapsed_ms.store(0);
        bucket.outputs.store(0);
        bucket.macs.store(0);
    }
    acoustic_packed_profile_reset();
    acoustic_phase_profile_reset();
}

static void acoustic_profile_add(
    const char* label,
    uint32_t elapsed_ms,
    uint64_t outputs,
    uint64_t macs
) {
    for (auto& bucket : g_acoustic_profile_buckets) {
        if (std::strcmp(bucket.label, label) == 0) {
            bucket.calls.fetch_add(1);
            bucket.elapsed_ms.fetch_add(elapsed_ms);
            bucket.outputs.fetch_add(outputs);
            bucket.macs.fetch_add(macs);
            return;
        }
    }
}

static void acoustic_profile_log(
    const char* stage,
    uint32_t graph_compute_ms,
    const ggml_cgraph* graph
) {
    uint64_t custom_ms = 0;
    uint64_t dense_ms = 0;
    uint64_t dense_outputs = 0;
    uint64_t dense_macs = 0;
    for (const auto& bucket : g_acoustic_profile_buckets) {
        const uint64_t elapsed_ms = bucket.elapsed_ms.load();
        custom_ms += elapsed_ms;
        if (std::strncmp(bucket.label, "dense.", 6) == 0) {
            dense_ms += elapsed_ms;
            dense_outputs += bucket.outputs.load();
            dense_macs += bucket.macs.load();
        }
    }

    uint32_t mul_mat_calls = 0;
    uint32_t custom_calls = 0;
    uint32_t conv_calls = 0;
    uint32_t unary_calls = 0;
    uint32_t cont_calls = 0;
    uint32_t permute_calls = 0;
    uint64_t mul_mat_macs = 0;
    if (graph != nullptr) {
        for (int i = 0; i < ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph)); i++) {
            const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
            if (node == nullptr) {
                continue;
            }
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    mul_mat_calls++;
                    if (node->src[0] != nullptr) {
                        mul_mat_macs += (uint64_t)ggml_nelements(node) * (uint64_t)node->src[0]->ne[0];
                    }
                    break;
                case GGML_OP_MAP_CUSTOM1:
                case GGML_OP_MAP_CUSTOM2:
                case GGML_OP_MAP_CUSTOM3:
                case GGML_OP_CUSTOM:
                    custom_calls++;
                    break;
                case GGML_OP_IM2COL:
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_2D_DW:
                    conv_calls++;
                    break;
                case GGML_OP_UNARY:
                    unary_calls++;
                    break;
                case GGML_OP_CONT:
                    cont_calls++;
                    break;
                case GGML_OP_PERMUTE:
                    permute_calls++;
                    break;
                default:
                    break;
            }
        }
    }

    const int64_t graph_minus_custom_worker_ms =
        static_cast<int64_t>(graph_compute_ms) -
        static_cast<int64_t>(custom_ms);
    fprintf(stderr,
            "[AcousticProfile] stage=%s graph_ms=%u custom_worker_ms=%llu "
            "graph_minus_custom_worker_ms=%lld "
            "nodes=%d mul_mat_calls=%u mul_mat_macs=%llu custom_calls=%u "
            "conv_nodes=%u unary_nodes=%u cont_nodes=%u permute_nodes=%u skip_postnet=%d\n",
            stage,
            (unsigned)graph_compute_ms,
            (unsigned long long)custom_ms,
            (long long)graph_minus_custom_worker_ms,
            graph != nullptr ? ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph)) : 0,
            (unsigned)mul_mat_calls,
            (unsigned long long)mul_mat_macs,
            (unsigned)custom_calls,
            (unsigned)conv_calls,
            (unsigned)unary_calls,
            (unsigned)cont_calls,
            (unsigned)permute_calls,
            INFLECT_ACOUSTIC_SKIP_POSTNET);

    if (dense_outputs != 0) {
        fprintf(stderr,
                "[AcousticProfile] family=dense_linear worker_ms=%llu "
                "outputs=%llu macs=%llu ns_per_mac=%llu\n",
                (unsigned long long)dense_ms,
                (unsigned long long)dense_outputs,
                (unsigned long long)dense_macs,
                dense_macs > 0
                    ? (unsigned long long)(
                          (dense_ms * 1000000ULL) /
                          dense_macs)
                    : 0ULL);
    }

    for (const auto& bucket : g_acoustic_profile_buckets) {
        const uint32_t calls = bucket.calls.load();
        if (calls == 0) {
            continue;
        }
        const uint64_t elapsed_ms = bucket.elapsed_ms.load();
        const uint64_t outputs = bucket.outputs.load();
        const uint64_t macs = bucket.macs.load();
        fprintf(stderr,
                "[AcousticProfile] op=%s calls=%u ms=%llu outputs=%llu macs=%llu us_per_output=%llu ns_per_mac=%llu\n",
                bucket.label,
                (unsigned)calls,
                (unsigned long long)elapsed_ms,
                (unsigned long long)outputs,
                (unsigned long long)macs,
                outputs > 0 ? (unsigned long long)((elapsed_ms * 1000ULL) / outputs) : 0ULL,
                macs > 0 ? (unsigned long long)((elapsed_ms * 1000000ULL) / macs) : 0ULL);
    }
#if defined(INFLECT_LOW_MEMORY)
    for (const auto& profile : g_acoustic_packed_profiles) {
        if (profile.input_rows.load() == 0 &&
            profile.output_rows.load() == 0) {
            continue;
        }
        const uint64_t total_cycles =
            profile.total_cycles.load();
        const uint64_t classified_cycles =
            profile.scratch_init_cycles.load() +
            profile.cooperate_cycles.load() +
            profile.input_quant_cycles.load() +
            profile.q4_unpack_cycles.load() +
            profile.s8_dot_cycles.load() +
            profile.scale_reduce_cycles.load() +
            profile.output_write_cycles.load();
        const uint64_t unclassified_cycles =
            total_cycles > classified_cycles
                ? total_cycles - classified_cycles
                : 0;
        fprintf(stderr,
                "[AcousticPackedProfile] stage=%s family=%s tile=%d "
                "input_rows=%llu output_rows=%llu total_cycles=%llu "
                "scratch_init_cycles=%llu cooperate_cycles=%llu "
                "input_quant_cycles=%llu "
                "q4_unpack_cycles=%llu s8_dot_cycles=%llu "
                "scale_reduce_cycles=%llu output_write_cycles=%llu "
                "unclassified_cycles=%llu\n",
                stage,
                profile.label,
                runtime_packed_quant_time_tile(),
                (unsigned long long)profile.input_rows.load(),
                (unsigned long long)profile.output_rows.load(),
                (unsigned long long)total_cycles,
                (unsigned long long)
                    profile.scratch_init_cycles.load(),
                (unsigned long long)
                    profile.cooperate_cycles.load(),
                (unsigned long long)profile.input_quant_cycles.load(),
                (unsigned long long)profile.q4_unpack_cycles.load(),
                (unsigned long long)profile.s8_dot_cycles.load(),
                (unsigned long long)profile.scale_reduce_cycles.load(),
                (unsigned long long)profile.output_write_cycles.load(),
                (unsigned long long)unclassified_cycles);
    }
#endif
    for (const auto& bucket :
         g_acoustic_phase_profile_buckets) {
        const uint64_t units = bucket.units.load();
        if (units == 0) {
            continue;
        }
        const uint64_t cycles = bucket.cycles.load();
        fprintf(stderr,
                "[AcousticPhaseProfile] stage=%s phase=%s cycles=%llu "
                "units=%llu cycles_per_unit=%llu sample_stride=%u\n",
                stage,
                bucket.label,
                (unsigned long long)cycles,
                (unsigned long long)units,
                (unsigned long long)(cycles / units),
                (unsigned)bucket.sample_stride);
    }
}
#else
#define ACOUSTIC_PROFILE_START_MS(name) ((void)0)
#define acoustic_profile_reset() ((void)0)
#define acoustic_profile_add(...) ((void)0)
#define acoustic_profile_log(...) ((void)0)
#define acoustic_phase_profile_add(...) ((void)0)
#if defined(INFLECT_LOW_MEMORY)
#define acoustic_packed_profile_add(...) ((void)0)
#endif
#endif

static void print_tensor_shape(const char* label, const ggml_tensor* t) {
    fprintf(stderr, "%s %s: [%lld, %lld, %lld, %lld]\n",
            label,
            t && t->name[0] ? t->name : "(unnamed)",
            t ? (long long)t->ne[0] : 0,
            t ? (long long)t->ne[1] : 0,
            t ? (long long)t->ne[2] : 0,
            t ? (long long)t->ne[3] : 0);
}

static bool debug_dump_enabled() {
    const char* dir = std::getenv("INFLECT_DUMP_DIR");
    return dir && dir[0];
}

static bool reserve_and_alloc_graph(ggml_gallocr_t allocr, ggml_cgraph* graph, const char* label) {
    if (!ggml_gallocr_reserve(allocr, graph) || !ggml_gallocr_alloc_graph(allocr, graph)) {
        fprintf(stderr, "[AcousticModel] Failed to allocate %s graph\n", label);
        return false;
    }
    return true;
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
            fprintf(stderr, "[AcousticModel] unsupported quantized tensor read type %s for %s\n",
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
    fprintf(stderr, "[AcousticModel] unsupported direct tensor read type %s for %s\n",
            ggml_type_name(t->type), t->name);
    std::abort();
}

static void tensor_row_to_f32(const ggml_tensor* t, int64_t i1, int64_t i2, FloatScratch& out) {
    out.resize(t->ne[0]);
    const char* row_ptr = (const char*)t->data + i1 * t->nb[1] + i2 * t->nb[2];
    if (t->type == GGML_TYPE_F32) {
        for (int64_t i = 0; i < t->ne[0]; i++) {
            out[i] = *reinterpret_cast<const float*>(row_ptr + i * t->nb[0]);
        }
        return;
    }
    if (t->type == GGML_TYPE_F16) {
        for (int64_t i = 0; i < t->ne[0]; i++) {
            out[i] = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(row_ptr + i * t->nb[0]));
        }
        return;
    }
    if (ggml_is_quantized(t->type)) {
        const auto* traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            fprintf(stderr, "[AcousticModel] unsupported quantized tensor row type %s for %s\n",
                    ggml_type_name(t->type), t->name);
            std::abort();
        }
        traits->to_float(row_ptr, out.data(), t->ne[0]);
        return;
    }
    fprintf(stderr, "[AcousticModel] unsupported tensor row type %s for %s\n",
            ggml_type_name(t->type), t->name);
    std::abort();
}

static void tensor_rows_to_f32(const ggml_tensor* t, FloatScratch& out) {
    const int64_t width = t->ne[0];
    const int64_t rows = t->ne[1] * t->ne[2];
    out.resize((size_t)(width * rows));
    thread_local FloatScratch row;
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
        for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
            tensor_row_to_f32(t, i1, i2, row);
            std::memcpy(out.data() + (size_t)((i2 * t->ne[1] + i1) * width),
                        row.data(),
                        (size_t)width * sizeof(float));
        }
    }
}

#if defined(INFLECT_LOW_MEMORY)
class AcousticQ4BatchDot {
public:
    bool init(int64_t elements, int rows) {
        if (elements <= 0 ||
            elements % kAcousticQ4BlockElements != 0 ||
            elements > INT32_MAX || rows <= 0) {
            return false;
        }
        elements_ = static_cast<int32_t>(elements);
        blocks_ = elements_ / kAcousticQ4BlockElements;
        rows_ = rows;
        const size_t element_count =
            static_cast<size_t>(elements_);
        const size_t block_count =
            static_cast<size_t>(blocks_);
        const size_t row_count =
            static_cast<size_t>(rows_);
        return input_values_.try_resize(
                   element_count * row_count) &&
               input_scales_.try_resize(
                   block_count * row_count * sizeof(float)) &&
               weight_values_.try_resize(element_count) &&
               weight_scale_bits_.try_resize(
                   block_count * sizeof(uint16_t)) &&
               weight_scales_.try_resize(
                   block_count * sizeof(float)) &&
               sums_.try_resize(
                   block_count * row_count * sizeof(int32_t)) &&
               results_.try_resize(row_count * sizeof(float)) &&
               padded_input_.try_resize(
                   element_count * sizeof(float));
    }

    void quantize_inputs(
        const float* input,
        int input_elements,
        int input_stride,
        int rows
    ) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t started = runtime_now_cycles();
#endif
        auto* quantized =
            static_cast<int8_t*>(input_values_.data());
        auto* scales =
            static_cast<float*>(input_scales_.data());
        auto* padded =
            static_cast<float*>(padded_input_.data());
        QuantizeF32ToQ8Blocks32Fn quantize =
            runtime_config().quantize_f32_to_q8_blocks_32;

        for (int row = 0; row < rows; ++row) {
            const float* source =
                input + static_cast<size_t>(row) *
                    static_cast<size_t>(input_stride);
            if (input_elements != elements_) {
                std::copy(
                    source, source + input_elements, padded);
                std::fill(
                    padded + input_elements,
                    padded + elements_,
                    0.0f);
                source = padded;
            }
            int8_t* row_values =
                quantized + static_cast<size_t>(row) *
                    static_cast<size_t>(elements_);
            float* row_scales =
                scales + static_cast<size_t>(row) *
                    static_cast<size_t>(blocks_);
            if (quantize != nullptr) {
                quantize(
                    source,
                    row_values,
                    row_scales,
                    static_cast<size_t>(blocks_),
                    false,
                    nullptr,
                    nullptr,
                    nullptr);
            } else {
                runtime_quantize_f32_to_q8_blocks_32(
                    source,
                    row_values,
                    row_scales,
                    static_cast<size_t>(blocks_),
                    false,
                    nullptr,
                    nullptr,
                    nullptr);
            }
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        phases_.input_quant_cycles +=
            static_cast<uint32_t>(
                runtime_now_cycles() - started);
        phases_.input_rows += static_cast<uint64_t>(rows);
#endif
    }

    void unpack_weight(const void* row) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t started = runtime_now_cycles();
#endif
        auto* values =
            static_cast<int8_t*>(weight_values_.data());
        auto* scale_bits =
            static_cast<uint16_t*>(
                weight_scale_bits_.data());
        auto* scales =
            static_cast<float*>(weight_scales_.data());
        runtime_unpack_q4_0_blocks_32(
            static_cast<const uint8_t*>(row),
            sizeof(AcousticPackedQ4Block),
            values,
            scale_bits,
            static_cast<size_t>(blocks_));
        for (int block = 0; block < blocks_; ++block) {
            scales[block] =
                ggml_fp16_to_fp32(scale_bits[block]);
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        phases_.q4_unpack_cycles +=
            static_cast<uint32_t>(
                runtime_now_cycles() - started);
#endif
    }

    bool prepare_weights(const ggml_tensor* weight) {
        if (weight == nullptr ||
            weight->type != GGML_TYPE_Q4_0 ||
            weight->ne[0] != elements_ ||
            weight->ne[1] <= 0) {
            return false;
        }
        prepared_rows_ = static_cast<int32_t>(weight->ne[1]);
        const size_t prepared_values =
            static_cast<size_t>(prepared_rows_) *
            static_cast<size_t>(elements_);
        const size_t prepared_scales =
            static_cast<size_t>(prepared_rows_) *
            static_cast<size_t>(blocks_);
        if (!prepared_weight_values_.try_resize(
                prepared_values) ||
            !prepared_weight_scales_.try_resize(
                prepared_scales * sizeof(float))) {
            prepared_rows_ = 0;
            return false;
        }

        auto* values = static_cast<int8_t*>(
            prepared_weight_values_.data());
        auto* scales = static_cast<float*>(
            prepared_weight_scales_.data());
        for (int32_t output = 0;
             output < prepared_rows_;
             ++output) {
            const char* row =
                static_cast<const char*>(weight->data) +
                static_cast<size_t>(output) *
                    weight->nb[1];
            unpack_weight(row);
            std::memcpy(
                values +
                    static_cast<size_t>(output) *
                        static_cast<size_t>(elements_),
                weight_values_.data(),
                static_cast<size_t>(elements_));
            std::memcpy(
                scales +
                    static_cast<size_t>(output) *
                        static_cast<size_t>(blocks_),
                weight_scales_.data(),
                static_cast<size_t>(blocks_) *
                    sizeof(float));
        }
        return true;
    }

    void calculate(int rows) {
        calculate_with_weights(
            static_cast<const int8_t*>(
                weight_values_.data()),
            static_cast<const float*>(
                weight_scales_.data()),
            rows);
    }

    void calculate_prepared(int output_channel, int rows) {
        if (output_channel < 0 ||
            output_channel >= prepared_rows_) {
            fprintf(stderr,
                    "[AcousticModel] invalid prepared Q4 output row\n");
            std::abort();
        }
        calculate_with_weights(
            static_cast<const int8_t*>(
                prepared_weight_values_.data()) +
                static_cast<size_t>(output_channel) *
                    static_cast<size_t>(elements_),
            static_cast<const float*>(
                prepared_weight_scales_.data()) +
                static_cast<size_t>(output_channel) *
                    static_cast<size_t>(blocks_),
            rows);
    }

    float result(int row) const {
        return static_cast<const float*>(
            results_.data())[row];
    }

    void write_results(
        float* output,
        int output_row_stride,
        int output_channel_stride,
        int output_channel,
        float bias,
        int rows
    ) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t started = runtime_now_cycles();
#endif
        const auto* results =
            static_cast<const float*>(results_.data());
        for (int row = 0; row < rows; ++row) {
            output[
                static_cast<size_t>(row) *
                    static_cast<size_t>(output_row_stride) +
                static_cast<size_t>(output_channel) *
                    static_cast<size_t>(
                        output_channel_stride)] =
                    results[row] + bias;
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        phases_.output_write_cycles +=
            static_cast<uint32_t>(
                runtime_now_cycles() - started);
        phases_.output_rows += static_cast<uint64_t>(rows);
#endif
    }

#if INFLECT_PROFILE_ACOUSTIC_OPS
    const AcousticPackedPhaseTimes& phases() const {
        return phases_;
    }
#endif

private:
    void calculate_with_weights(
        const int8_t* weight_values,
        const float* weight_scales,
        int rows
    ) {
        if (runtime_has_s8_scaled_dot_blocks_32()) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
            uint64_t dot_cycles = 0;
            uint64_t scale_cycles = 0;
            uint64_t* dot_cycles_ptr = &dot_cycles;
            uint64_t* scale_cycles_ptr = &scale_cycles;
#else
            uint64_t* dot_cycles_ptr = nullptr;
            uint64_t* scale_cycles_ptr = nullptr;
#endif
            runtime_dot_s8_scaled_blocks_32(
                weight_values,
                weight_scales,
                static_cast<const int8_t*>(
                    input_values_.data()),
                static_cast<const float*>(
                    input_scales_.data()),
                static_cast<float*>(results_.data()),
                static_cast<size_t>(blocks_),
                static_cast<size_t>(rows),
                false,
                dot_cycles_ptr,
                scale_cycles_ptr);
#if INFLECT_PROFILE_ACOUSTIC_OPS
            phases_.s8_dot_cycles += dot_cycles;
            phases_.scale_reduce_cycles += scale_cycles;
#endif
            return;
        }

#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t dot_started = runtime_now_cycles();
#endif
        runtime_dot_s8_blocks_32(
            weight_values,
            static_cast<const int8_t*>(
                input_values_.data()),
            static_cast<int32_t*>(sums_.data()),
            static_cast<size_t>(blocks_),
            static_cast<size_t>(rows));
#if INFLECT_PROFILE_ACOUSTIC_OPS
        phases_.s8_dot_cycles +=
            static_cast<uint32_t>(
                runtime_now_cycles() - dot_started);
        const uint32_t reduce_started =
            runtime_now_cycles();
#endif
        const auto* products =
            static_cast<const int32_t*>(sums_.data());
        const auto* input_scales =
            static_cast<const float*>(input_scales_.data());
        auto* results =
            static_cast<float*>(results_.data());
        for (int row = 0; row < rows; ++row) {
            float value = 0.0f;
            for (int block = 0; block < blocks_; ++block) {
                const size_t index =
                    static_cast<size_t>(row) *
                        static_cast<size_t>(blocks_) +
                    static_cast<size_t>(block);
                value += static_cast<float>(products[index]) *
                         weight_scales[block] *
                         input_scales[index];
            }
            results[row] = value;
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        phases_.scale_reduce_cycles +=
            static_cast<uint32_t>(
                runtime_now_cycles() - reduce_started);
#endif
    }

    int32_t elements_ = 0;
    int32_t blocks_ = 0;
    int32_t rows_ = 0;
    int32_t prepared_rows_ = 0;
    AcousticInternalScratch input_values_;
    AcousticInternalScratch input_scales_;
    AcousticInternalScratch weight_values_;
    AcousticInternalScratch weight_scale_bits_;
    AcousticInternalScratch weight_scales_;
    AcousticInternalScratch sums_;
    AcousticInternalScratch results_;
    AcousticInternalScratch padded_input_;
    AcousticInternalScratch prepared_weight_values_;
    AcousticInternalScratch prepared_weight_scales_;
#if INFLECT_PROFILE_ACOUSTIC_OPS
    AcousticPackedPhaseTimes phases_;
#endif
};

static bool acoustic_q4_linear_batch(
    AcousticPackedProfileKind profile_kind,
    const ggml_tensor* weight,
    const ggml_tensor* bias,
    const float* input,
    int input_elements,
    int input_stride,
    int rows,
    float* output,
    int output_row_stride,
    int output_channel_stride
) {
    (void)profile_kind;
    if (weight == nullptr || weight->type != GGML_TYPE_Q4_0 ||
        input == nullptr || output == nullptr ||
        input_elements <= 0 || rows <= 0 ||
        weight->ne[0] < input_elements ||
        weight->ne[0] % kAcousticQ4BlockElements != 0 ||
        weight->ne[1] <= 0) {
        return false;
    }

    const int tile_capacity =
        std::max(1, runtime_packed_quant_time_tile());
#if INFLECT_PROFILE_ACOUSTIC_OPS
    const uint32_t total_started = runtime_now_cycles();
    uint64_t scratch_init_cycles = 0;
    uint64_t cooperate_cycles = 0;
    AcousticPackedPhaseTimes completed_phases;
#endif
    {
        AcousticQ4BatchDot dot;
#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t scratch_init_started =
            runtime_now_cycles();
#endif
        if (!dot.init(weight->ne[0], tile_capacity)) {
            return false;
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        scratch_init_cycles += static_cast<uint32_t>(
            runtime_now_cycles() - scratch_init_started);
#endif

        for (int first = 0; first < rows;
             first += tile_capacity) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
            const uint32_t cooperate_started =
                runtime_now_cycles();
#endif
            runtime_cooperate();
#if INFLECT_PROFILE_ACOUSTIC_OPS
            cooperate_cycles += static_cast<uint32_t>(
                runtime_now_cycles() - cooperate_started);
#endif
            const int tile_rows =
                std::min(tile_capacity, rows - first);
            dot.quantize_inputs(
                input + static_cast<size_t>(first) *
                    static_cast<size_t>(input_stride),
                input_elements,
                input_stride,
                tile_rows);
            for (int output_channel = 0;
                 output_channel < weight->ne[1];
                 ++output_channel) {
                const char* weight_row =
                    static_cast<const char*>(weight->data) +
                    static_cast<size_t>(output_channel) *
                        weight->nb[1];
                dot.unpack_weight(weight_row);
                dot.calculate(tile_rows);
                const float bias_value =
                    bias != nullptr
                        ? tensor_get_f32(
                              bias, output_channel, 0, 0)
                        : 0.0f;
                dot.write_results(
                    output +
                        static_cast<size_t>(first) *
                            static_cast<size_t>(
                                output_row_stride),
                    output_row_stride,
                    output_channel_stride,
                    output_channel,
                    bias_value,
                    tile_rows);
            }
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        completed_phases = dot.phases();
#endif
    }
#if INFLECT_PROFILE_ACOUSTIC_OPS
    completed_phases.total_cycles +=
        static_cast<uint32_t>(
            runtime_now_cycles() - total_started);
    completed_phases.scratch_init_cycles +=
        scratch_init_cycles;
    completed_phases.cooperate_cycles +=
        cooperate_cycles;
    acoustic_packed_profile_add(
        profile_kind, completed_phases);
#endif
    return true;
}
#endif

static float dot_f32(const float* a, const float* b, int n) {
    float out = 0.0f;
#if INFLECT_HAS_ESP_DSP && INFLECT_USE_ESP_DSP_CONTIG
    if (dsps_dotprod_f32(a, b, &out, n) == ESP_OK) {
        return out;
    }
#endif
    int i = 0;
    for (; i + 3 < n; i += 4) {
        out += a[i] * b[i] + a[i + 1] * b[i + 1] +
               a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) {
        out += a[i] * b[i];
    }
    return out;
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

static void tensor_set_f32(ggml_tensor* t, float v, int64_t i0, int64_t i1, int64_t i2) {
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "[AcousticModel] unsupported direct tensor write type %s for %s\n",
                ggml_type_name(t->type), t->name);
        std::abort();
    }
    *(float*)((char*)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2]) = v;
}

// ── Debug: dump a tensor's first few values and stats ───────────────────
void dump_tensor(const ggml_tensor* t, const char* label) {
    if (!t || !t->data) { fprintf(stderr, "[dump] %s: null\n", label); return; }
    int64_t n = ggml_nelements(t);
    int64_t n0 = t->ne[0], n1 = t->ne[1];
    float mn = 1e30f, mx = -1e30f, sum = 0;
    // iterate as flat for stats
    for (int64_t i = 0; i < n && i < 100000; i++) {
        float v = tensor_get_f32(t, i % n0, (i / n0) % (n1 > 0 ? n1 : 1), i / (n0 * (n1 > 0 ? n1 : 1)));
        mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
    }
    fprintf(stderr, "[dump] %s (%ld elements) first5=[", label, (long)n);
    for (int64_t i = 0; i < 5 && i < n; i++) {
        float v = tensor_get_f32(t, i % n0, (i / n0) % (n1 > 0 ? n1 : 1), i / (n0 * (n1 > 0 ? n1 : 1)));
        fprintf(stderr, "%.4f%c", v, i < 4 ? ',' : ']');
    }
    fprintf(stderr, " min=%.4f max=%.4f mean=%.4f\n", mn, mx, sum / std::min(n, (int64_t)100000));
}

// ═════════════════════════════════════════════════════════════════════════
// Utility: require dimensions compatible for broadcasting add
// ═════════════════════════════════════════════════════════════════════════

static void require_repeat_compatible(const char* op, const char* where,
                                       const ggml_tensor* a, const ggml_tensor* b) {
    for (int i = 0; i < 4; i++) {
        if (a->ne[i] != b->ne[i] && b->ne[i] != 1 && a->ne[i] != 1) {
            fprintf(stderr, "[AcousticModel] %s %s: shape mismatch a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]\n",
                    op, where,
                    (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
                    (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
            std::abort();
        }
    }
    // Also check that broadcasting is compatible: larger dim % smaller dim == 0
    for (int i = 0; i < 4; i++) {
        int64_t max_dim = std::max(a->ne[i], b->ne[i]);
        int64_t min_dim = std::min(a->ne[i], b->ne[i]);
        if (max_dim != min_dim && (max_dim % min_dim != 0)) {
            fprintf(stderr, "[AcousticModel] %s %s: broadcast incompatible for dim %d: %lld vs %lld\n",
                    op, where, i, (long long)a->ne[i], (long long)b->ne[i]);
            std::abort();
        }
    }
}
static ggml_tensor* checked_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, const char* where) {
    require_repeat_compatible("add", where, a, b);
    if (b->type != a->type && !ggml_is_quantized(b->type)) {
        b = ggml_cast(ctx, b, a->type);
    }
    return ggml_add(ctx, a, b);
}

static ggml_tensor* trim_ne0(ggml_context* ctx, ggml_tensor* x, int64_t ne0) {
    if (x->ne[0] == ne0) {
        return x;
    }
    if (x->ne[0] < ne0 || x->ne[3] != 1) {
        fprintf(stderr, "[AcousticModel] cannot trim tensor %s from ne0=%lld to %lld\n",
                x->name, (long long)x->ne[0], (long long)ne0);
        std::abort();
    }
    return ggml_view_3d(ctx, x, ne0, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
}

static void quant_conv1d_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata
) {
    ACOUSTIC_PROFILE_START_MS(op_start_ms);
    const auto* p = static_cast<const QuantConv1dOpData*>(userdata);
    const ggml_tensor* x = dst->src[0];      // [T, in_ch, B]
    const ggml_tensor* weight = dst->src[1]; // [K*in_ch padded, out_ch]
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_quantized(weight->type)) {
        fprintf(stderr, "[AcousticModel] unsupported low-memory conv1d tensor types\n");
        std::abort();
    }

    const int64_t T = dst->ne[0];
    const int64_t out_ch = dst->ne[1];
    const int64_t batch = dst->ne[2];
    const int64_t in_ch = x->ne[1];

#if defined(INFLECT_LOW_MEMORY)
    if (weight->type == GGML_TYPE_Q4_0) {
        const int64_t flat =
            p->kernel_size * in_ch;
        const int64_t total_rows = T * batch;
        const int64_t first = (total_rows * ith) / nth;
        const int64_t end =
            (total_rows * (ith + 1)) / nth;
        const int rows = static_cast<int>(end - first);
        if (rows > 0) {
            FloatScratch gathered;
            gathered.resize(
                static_cast<size_t>(rows) *
                static_cast<size_t>(flat));
            for (int row = 0; row < rows; ++row) {
                const int64_t source_row = first + row;
                const int64_t b = source_row / T;
                const int64_t t = source_row % T;
                float* destination =
                    gathered.data() +
                    static_cast<size_t>(row) *
                        static_cast<size_t>(flat);
                for (int64_t k = 0;
                     k < p->kernel_size;
                     ++k) {
                    const int64_t source_t =
                        t * p->stride +
                        k * p->dilation -
                        p->padding;
                    float* kernel_values =
                        destination +
                        static_cast<size_t>(k) *
                            static_cast<size_t>(in_ch);
                    if (source_t < 0 ||
                        source_t >= x->ne[0]) {
                        std::fill(
                            kernel_values,
                            kernel_values + in_ch,
                            0.0f);
                        continue;
                    }
                    const char* source =
                        static_cast<const char*>(x->data) +
                        source_t * x->nb[0] +
                        b * x->nb[2];
                    for (int64_t channel = 0;
                         channel < in_ch;
                         ++channel) {
                        kernel_values[channel] =
                            *reinterpret_cast<const float*>(
                                source +
                                channel * x->nb[1]);
                    }
                }
            }
            const int64_t first_batch = first / T;
            const int64_t first_time = first % T;
            // Worker row partitions never cross a batch in the current
            // batch-one TTS graph. Keep the invariant explicit because
            // output time is the contiguous dimension here.
            if (first_time + rows > T) {
                fprintf(stderr,
                        "[AcousticModel] packed postnet batch partition unsupported\n");
                std::abort();
            }
            float* output =
                reinterpret_cast<float*>(
                    static_cast<char*>(dst->data) +
                    first_time * dst->nb[0] +
                    first_batch * dst->nb[2]);
            if (!acoustic_q4_linear_batch(
                    AcousticPackedProfileKind::PostnetConv1d,
                    weight,
                    nullptr,
                    gathered.data(),
                    static_cast<int>(flat),
                    static_cast<int>(flat),
                    rows,
                    output,
                    static_cast<int>(
                        dst->nb[0] / sizeof(float)),
                    static_cast<int>(
                        dst->nb[1] / sizeof(float)))) {
                fprintf(stderr,
                        "[AcousticModel] packed postnet scratch allocation failed\n");
                std::abort();
            }
        }
        const uint64_t outputs =
            static_cast<uint64_t>(rows) *
            static_cast<uint64_t>(out_ch);
        (void)outputs;
        acoustic_profile_add(
            "postnet.conv1d",
            runtime_now_ms() - op_start_ms,
            outputs,
            outputs *
                static_cast<uint64_t>(
                    p->kernel_size * in_ch));
        return;
    }
#endif

    const auto* traits = ggml_get_type_traits(weight->type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "[AcousticModel] unsupported quantized conv1d weight type %s\n",
                ggml_type_name(weight->type));
        std::abort();
    }

    thread_local FloatScratch weight_row;
    weight_row.resize(weight->ne[0]);

    const char* x_data = static_cast<const char*>(x->data);
    char* dst_data = static_cast<char*>(dst->data);

    const int64_t o_start = (out_ch * ith) / nth;
    const int64_t o_end = (out_ch * (ith + 1)) / nth;
    for (int64_t o = o_start; o < o_end; o++) {
        const char* row_ptr = static_cast<const char*>(weight->data) + o * weight->nb[1];
        traits->to_float(row_ptr, weight_row.data(), weight->ne[0]);

        for (int64_t b = 0; b < batch; b++) {
            for (int64_t t = 0; t < T; t++) {
                float v = 0.0f;
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
            }
        }
    }

    const uint64_t outputs = (uint64_t)(o_end - o_start) * (uint64_t)T * (uint64_t)batch;
    const uint64_t macs = outputs * (uint64_t)p->kernel_size * (uint64_t)in_ch;
    (void)outputs;
    (void)macs;
    acoustic_profile_add(
        "postnet.conv1d",
        runtime_now_ms() - op_start_ms,
        outputs,
        macs);
}

#if 0
#if defined(INFLECT_LOW_MEMORY)
static void quant_postnet_conv1d_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata
) {
    ACOUSTIC_PROFILE_START_MS(op_start_ms);
    const auto* p =
        static_cast<const QuantPostnetConv1dOpData*>(
            userdata);
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* weight = dst->src[1];
    const ggml_tensor* bias = dst->src[2];
    const ggml_tensor* residual =
        p->add_residual ? dst->src[3] : nullptr;

    if (x == nullptr || weight == nullptr ||
        bias == nullptr ||
        x->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 ||
        weight->type != GGML_TYPE_Q4_0 ||
        x->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float) ||
        (residual != nullptr &&
         (residual->type != GGML_TYPE_F32 ||
          residual->nb[0] != sizeof(float)))) {
        fprintf(
            stderr,
            "[AcousticModel] unsupported packed postnet tensors\n");
        std::abort();
    }

    // Keep postnet tensors in acoustic [channel, time, batch]
    // layout. A complete input frame is contiguous, so each
    // kernel tap becomes one memcpy instead of a strided gather.
    const int64_t in_ch = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t batch = x->ne[2];
    const int64_t out_ch = dst->ne[0];
    const int64_t flat = p->kernel_size * in_ch;
    const int64_t total_rows = T * batch;
    const int64_t first = (total_rows * ith) / nth;
    const int64_t end =
        (total_rows * (ith + 1)) / nth;
    const int rows = static_cast<int>(end - first);

    if (weight->ne[0] < flat ||
        weight->ne[1] != out_ch ||
        dst->ne[1] != T ||
        dst->ne[2] != batch ||
        (residual != nullptr &&
         (residual->ne[0] != out_ch ||
          residual->ne[1] != T ||
          residual->ne[2] != batch))) {
        fprintf(
            stderr,
            "[AcousticModel] invalid packed postnet shapes\n");
        std::abort();
    }

    if (rows > 0) {
        FloatScratch gathered;
        gathered.resize(
            static_cast<size_t>(rows) *
            static_cast<size_t>(flat));
        for (int row = 0; row < rows; ++row) {
            const int64_t source_row = first + row;
            const int64_t b = source_row / T;
            const int64_t t = source_row % T;
            float* destination =
                gathered.data() +
                static_cast<size_t>(row) *
                    static_cast<size_t>(flat);
            for (int64_t k = 0;
                 k < p->kernel_size;
                 ++k) {
                const int64_t source_t =
                    t + k - p->padding;
                float* kernel_values =
                    destination +
                    static_cast<size_t>(k) *
                        static_cast<size_t>(in_ch);
                if (source_t < 0 || source_t >= T) {
                    std::fill(
                        kernel_values,
                        kernel_values + in_ch,
                        0.0f);
                    continue;
                }
                const char* source =
                    static_cast<const char*>(x->data) +
                    source_t * x->nb[1] +
                    b * x->nb[2];
                std::memcpy(
                    kernel_values,
                    source,
                    static_cast<size_t>(in_ch) *
                        sizeof(float));
            }
        }

        const int64_t first_batch = first / T;
        const int64_t first_time = first % T;
        if (first_time + rows > T) {
            fprintf(
                stderr,
                "[AcousticModel] packed postnet batch partition unsupported\n");
            std::abort();
        }
        auto* output =
            reinterpret_cast<float*>(
                static_cast<char*>(dst->data) +
                first_time * dst->nb[1] +
                first_batch * dst->nb[2]);
        const float* residual_data =
            residual != nullptr
                ? reinterpret_cast<const float*>(
                      static_cast<const char*>(
                          residual->data) +
                      first_time * residual->nb[1] +
                      first_batch * residual->nb[2])
                : nullptr;
        if (!acoustic_q4_linear_batch(
                AcousticPackedProfileKind::PostnetConv1d,
                weight,
                bias,
                gathered.data(),
                static_cast<int>(flat),
                static_cast<int>(flat),
                rows,
                output,
                static_cast<int>(
                    dst->nb[1] / sizeof(float)),
                1,
                p->apply_tanh
                    ? AcousticOutputActivation::Tanh
                    : AcousticOutputActivation::None,
                residual_data,
                residual != nullptr
                    ? static_cast<int>(
                          residual->nb[1] /
                          sizeof(float))
                    : 0,
                residual != nullptr ? 1 : 0,
                p->output_scale)) {
            fprintf(
                stderr,
                "[AcousticModel] packed postnet scratch allocation failed\n");
            std::abort();
        }
    }

    const uint64_t outputs =
        static_cast<uint64_t>(rows) *
        static_cast<uint64_t>(out_ch);
    acoustic_profile_add(
        "postnet.conv1d",
        runtime_now_ms() - op_start_ms,
        outputs,
        outputs * static_cast<uint64_t>(flat));
}

static ggml_tensor* quantized_postnet_conv1d(
    ggml_context* ctx,
    ggml_tensor* weight,
    ggml_tensor* x,
    ggml_tensor* bias,
    ggml_tensor* residual,
    int kernel_size,
    int padding,
    bool apply_tanh,
    float output_scale,
    std::vector<QuantPostnetConv1dOpData>& op_data
) {
    if (weight == nullptr || x == nullptr ||
        bias == nullptr ||
        weight->type != GGML_TYPE_Q4_0 ||
        x->type != GGML_TYPE_F32 ||
        weight->ne[0] < kernel_size * x->ne[0] ||
        weight->ne[1] <= 0 ||
        (residual != nullptr &&
         (residual->ne[0] != weight->ne[1] ||
          residual->ne[1] != x->ne[1] ||
          residual->ne[2] != x->ne[2]))) {
        return nullptr;
    }

    op_data.push_back({
        kernel_size,
        padding,
        apply_tanh,
        residual != nullptr,
        output_scale,
    });
    ggml_tensor* args[] = {
        x, weight, bias, residual,
    };
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        weight->ne[1],
        x->ne[1],
        x->ne[2],
        1,
        args,
        residual != nullptr ? 4 : 3,
        quant_postnet_conv1d_op,
        GGML_N_TASKS_MAX,
        &op_data.back());
}
#endif
#endif

static ggml_tensor* quant_or_f16_conv1d(
    ggml_context* ctx,
    ggml_tensor* weight,
    ggml_tensor* x,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    std::vector<QuantConv1dOpData>& op_data
) {
    if (!ggml_is_quantized(weight->type)) {
        ggml_tensor* kernel = weight;
        if (kernel->type != GGML_TYPE_F16) {
            kernel = ggml_cast(ctx, kernel, GGML_TYPE_F16);
        }
        return ggml_conv_1d(ctx, kernel, x, stride, padding, dilation);
    }

    const int64_t in_ch = x->ne[1];
    const int64_t out_ch = weight->ne[1];
    const int64_t flat = kernel_size * in_ch;
    if (weight->ne[0] < flat) {
        fprintf(stderr, "[AcousticModel] quantized conv weight too small: weight=[%lld,%lld] flat=%lld\n",
                (long long)weight->ne[0], (long long)weight->ne[1], (long long)flat);
        std::abort();
    }

    const int64_t out_t = (x->ne[0] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    op_data.push_back({kernel_size, stride, padding, dilation});
    ggml_tensor* args[] = {x, weight};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, out_t, out_ch, x->ne[2], 1,
                          args, 2, quant_conv1d_op, GGML_N_TASKS_MAX, &op_data.back());
}

// ═════════════════════════════════════════════════════════════════════════
// Layer helpers
// ═════════════════════════════════════════════════════════════════════════

static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* w, ggml_tensor* b) {
    auto layer_norm_op = [](
        ggml_tensor* dst,
        const ggml_tensor* src,
        const ggml_tensor* weight,
        const ggml_tensor* bias,
        int ith,
        int nth,
        void* userdata
    ) {
        (void)userdata;
        ACOUSTIC_PROFILE_START_MS(op_start_ms);
#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t phase_total_started =
            runtime_now_cycles();
        uint64_t mean_cycles = 0;
        uint64_t variance_cycles = 0;
        uint64_t affine_write_cycles = 0;
#endif
        const int64_t H = src->ne[0];
        const int64_t T = src->ne[1];
        const int64_t start = (T * ith) / nth;
        const int64_t end = (T * (ith + 1)) / nth;
        const bool direct_f32 =
            src->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 &&
            weight->type == GGML_TYPE_F32 &&
            bias->type == GGML_TYPE_F32 &&
            src->nb[0] == sizeof(float) &&
            dst->nb[0] == sizeof(float) &&
            weight->nb[0] == sizeof(float) &&
            bias->nb[0] == sizeof(float);
        const auto* source_data =
            static_cast<const char*>(src->data);
        auto* destination_data =
            static_cast<char*>(dst->data);
        const auto* weight_data =
            direct_f32
                ? static_cast<const float*>(weight->data)
                : nullptr;
        const auto* bias_data =
            direct_f32
                ? static_cast<const float*>(bias->data)
                : nullptr;
        for (int64_t t = start; t < end; t++) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
            const uint32_t mean_started =
                runtime_now_cycles();
#endif
            float mean = 0.0f;
            const float* source = nullptr;
            float* destination = nullptr;
            if (direct_f32) {
                source = reinterpret_cast<const float*>(
                    source_data + t * src->nb[1]);
                destination = reinterpret_cast<float*>(
                    destination_data + t * dst->nb[1]);
                for (int64_t h = 0; h < H; h++) {
                    mean += source[h];
                }
            } else {
                for (int64_t h = 0; h < H; h++) {
                    mean += tensor_get_f32(
                        src, h, t, 0);
                }
            }
            mean /= (float)H;

#if INFLECT_PROFILE_ACOUSTIC_OPS
            const uint32_t mean_finished =
                runtime_now_cycles();
#endif
            float var = 0.0f;
            if (direct_f32) {
                for (int64_t h = 0; h < H; h++) {
                    const float d = source[h] - mean;
                    var += d * d;
                }
            } else {
                for (int64_t h = 0; h < H; h++) {
                    const float d =
                        tensor_get_f32(
                            src, h, t, 0) -
                        mean;
                    var += d * d;
                }
            }
            const float inv_std = 1.0f / std::sqrt(var / (float)H + 1e-5f);

#if INFLECT_PROFILE_ACOUSTIC_OPS
            const uint32_t variance_finished =
                runtime_now_cycles();
#endif
            if (direct_f32) {
                for (int64_t h = 0; h < H; h++) {
                    const float normalized =
                        (source[h] - mean) * inv_std;
                    destination[h] =
                        normalized * weight_data[h] +
                        bias_data[h];
                }
            } else {
                for (int64_t h = 0; h < H; h++) {
                    const float normalized =
                        (tensor_get_f32(
                             src, h, t, 0) -
                         mean) *
                        inv_std;
                    const float v =
                        normalized *
                            tensor_get_f32(
                                weight, h, 0, 0) +
                        tensor_get_f32(
                            bias, h, 0, 0);
                    tensor_set_f32(
                        dst, v, h, t, 0);
                }
            }
#if INFLECT_PROFILE_ACOUSTIC_OPS
            const uint32_t affine_write_finished =
                runtime_now_cycles();
            mean_cycles += static_cast<uint32_t>(
                mean_finished - mean_started);
            variance_cycles += static_cast<uint32_t>(
                variance_finished - mean_finished);
            affine_write_cycles += static_cast<uint32_t>(
                affine_write_finished -
                variance_finished);
#endif
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint64_t phase_units =
            static_cast<uint64_t>(end - start) *
            static_cast<uint64_t>(H);
        acoustic_phase_profile_add(
            "layer_norm.total",
            static_cast<uint32_t>(
                runtime_now_cycles() -
                phase_total_started),
            phase_units);
        acoustic_phase_profile_add(
            "layer_norm.mean",
            mean_cycles,
            phase_units);
        acoustic_phase_profile_add(
            "layer_norm.variance",
            variance_cycles,
            phase_units);
        acoustic_phase_profile_add(
            "layer_norm.affine_write",
            affine_write_cycles,
            phase_units);
#endif
        acoustic_profile_add("layer_norm",
                             runtime_now_ms() - op_start_ms,
                             (uint64_t)(end - start) * (uint64_t)H,
                             0);
    };
    return ggml_map_custom3(ctx, x, w, b, layer_norm_op, GGML_N_TASKS_MAX, nullptr);
}

static ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                            ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* mat_w = w;
    if (w->ne[0] == 1 && w->ne[1] == x->ne[0]) {
        // Converted Conv1d(kernel=1) weights are stored as [K, in, out].
        // Reinterpret the K=1 slice as the [in, out] matrix expected by GGML.
        mat_w = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);
    }
    if (mat_w->ne[0] > x->ne[0]) {
        x = ggml_pad(ctx, x, mat_w->ne[0] - x->ne[0], 0, 0, 0);
    }
    if (x->ne[0] > mat_w->ne[0]) {
        x = trim_ne0(ctx, x, mat_w->ne[0]);
    }
    if (mat_w->ne[0] != x->ne[0]) {
        fprintf(stderr, "[AcousticModel] linear shape mismatch\n");
        print_tensor_shape("  w", mat_w);
        print_tensor_shape("  x", x);
        print_tensor_shape("  b", b);
        std::abort();
    }
    x = ggml_mul_mat(ctx, mat_w, x);
    if (b) {
        if (b->type != x->type && !ggml_is_quantized(b->type)) {
            b = ggml_cast(ctx, b, x->type);
        }
        x = ggml_add(ctx, x, b);
    }
    return x;
}

#if defined(INFLECT_LOW_MEMORY)
static void quant_linear_op(
    ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata
) {
    ACOUSTIC_PROFILE_START_MS(started_ms);
    const auto* data =
        static_cast<const QuantLinearOpData*>(userdata);
    const ggml_tensor* input = dst->src[0];
    const ggml_tensor* weight = dst->src[1];
    const ggml_tensor* bias =
        data->has_bias ? dst->src[2] : nullptr;
    if (input == nullptr || weight == nullptr ||
        input->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 ||
        weight->type != GGML_TYPE_Q4_0 ||
        input->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float)) {
        fprintf(stderr,
                "[AcousticModel] invalid packed linear tensors\n");
        std::abort();
    }

    const int64_t time = input->ne[1];
    const int64_t batch = input->ne[2] * input->ne[3];
    const int64_t total_rows = time * batch;
    const int64_t first = (total_rows * ith) / nth;
    const int64_t end = (total_rows * (ith + 1)) / nth;
    const int input_stride =
        static_cast<int>(input->nb[1] / sizeof(float));
    const int output_stride =
        static_cast<int>(dst->nb[1] / sizeof(float));
    const uint64_t output_count =
        static_cast<uint64_t>(end - first) *
        static_cast<uint64_t>(weight->ne[1]);
    (void)output_count;

    int64_t row = first;
    while (row < end) {
        const int64_t batch_index = row / time;
        const int64_t time_index = row % time;
        const int rows = static_cast<int>(
            std::min<int64_t>(end - row, time - time_index));
        const auto* input_data =
            reinterpret_cast<const float*>(
                static_cast<const char*>(input->data) +
                time_index * input->nb[1] +
                batch_index * input->nb[2]);
        auto* output_data =
            reinterpret_cast<float*>(
                static_cast<char*>(dst->data) +
                time_index * dst->nb[1] +
                batch_index * dst->nb[2]);
        if (!acoustic_q4_linear_batch(
                AcousticPackedProfileKind::DenseLinear,
                weight,
                bias,
                input_data,
                data->input_elements,
                input_stride,
                rows,
                output_data,
                output_stride,
                1)) {
            fprintf(stderr,
                    "[AcousticModel] packed linear scratch allocation failed\n");
            std::abort();
        }
        row += rows;
    }

    acoustic_profile_add(
        acoustic_dense_profile_label(weight),
        runtime_now_ms() - started_ms,
        output_count,
        output_count *
            static_cast<uint64_t>(data->input_elements));
}

static ggml_tensor* quantized_linear(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* weight,
    ggml_tensor* bias,
    std::vector<QuantLinearOpData>& op_data
) {
    constexpr uint64_t kMinimumPackedLinearMacs = 32768;
    if (input == nullptr || weight == nullptr ||
        input->type != GGML_TYPE_F32 ||
        weight->type != GGML_TYPE_Q4_0 ||
        input->ne[0] <= 0 ||
        weight->ne[0] < input->ne[0] ||
        weight->ne[0] %
                kAcousticQ4BlockElements !=
            0 ||
        weight->ne[1] <= 0) {
        return nullptr;
    }
    const uint64_t rows =
        static_cast<uint64_t>(input->ne[1]) *
        static_cast<uint64_t>(input->ne[2]) *
        static_cast<uint64_t>(input->ne[3]);
    const uint64_t macs =
        rows * static_cast<uint64_t>(input->ne[0]) *
        static_cast<uint64_t>(weight->ne[1]);
    if (macs < kMinimumPackedLinearMacs) {
        return nullptr;
    }

    op_data.push_back({
        static_cast<int>(input->ne[0]),
        bias != nullptr,
    });
    ggml_tensor* args[3] = {input, weight, bias};
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        weight->ne[1],
        input->ne[1],
        input->ne[2],
        input->ne[3],
        args,
        bias != nullptr ? 3 : 2,
        quant_linear_op,
        GGML_N_TASKS_MAX,
        &op_data.back());
}
#endif

// ═════════════════════════════════════════════════════════════════════════
// Depthwise Conv1d with gated activation
// ═════════════════════════════════════════════════════════════════════════

static void depthwise_gated_op(
    ggml_tensor* dst,
    const ggml_tensor* x,
    const ggml_tensor* weight,
    const ggml_tensor* bias,
    int ith,
    int nth,
    void* userdata
) {
    (void)userdata;
    ACOUSTIC_PROFILE_START_MS(op_start_ms);

    // x: [H, T, 1]  — channels = ne[0], time = ne[1]
    // weight: [K, H, 2]  — kernel, in_channel, filter+gate
    const int64_t H = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t K = weight->ne[0];
    const int64_t pad = K / 2;

    if (weight->ne[1] != H || weight->ne[2] != 2 || bias->ne[0] != 2 * H) {
        fprintf(stderr, "[AcousticModel] invalid depthwise gated tensor shapes\n");
        print_tensor_shape("  x     ", x);
        print_tensor_shape("  weight", weight);
        print_tensor_shape("  bias  ", bias);
        std::abort();
    }

    const int64_t total = H * T;
    const int64_t start = (total * ith) / nth;
    const int64_t end = (total * (ith + 1)) / nth;

    if (x->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 ||
        x->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float)) {
        fprintf(stderr,
                "[AcousticModel] unsupported depthwise tensor layout\n");
        std::abort();
    }

#if INFLECT_PROFILE_ACOUSTIC_OPS
    const uint32_t prepare_started =
        runtime_now_cycles();
    uint64_t bias_load_cycles = 0;
    uint64_t mac_cycles = 0;
    uint64_t gate_write_cycles = 0;
    uint64_t sampled_outputs = 0;
#endif
    FloatScratch weight_f;
    FloatScratch bias_f;
    tensor_rows_to_f32(weight, weight_f);
    tensor_rows_to_f32(bias, bias_f);
#if INFLECT_PROFILE_ACOUSTIC_OPS
    const uint64_t prepare_cycles =
        static_cast<uint32_t>(
            runtime_now_cycles() - prepare_started);
#endif

    const auto* x_data =
        static_cast<const char*>(x->data);
    auto* dst_data =
        static_cast<char*>(dst->data);
    for (int64_t idx = start; idx < end; idx++) {
        const int64_t h = idx % H;
        const int64_t t = idx / H;

        const int64_t a_oc = h;
        const int64_t b_oc = H + h;
        const int64_t out_per_group = 2;
        const int64_t a_src_h = a_oc / out_per_group;
        const int64_t b_src_h = b_oc / out_per_group;

#if INFLECT_PROFILE_ACOUSTIC_OPS
        const bool profile_sample =
            idx % kAcousticDepthwiseProfileStride == 0;
        const uint32_t bias_load_started =
            profile_sample ? runtime_now_cycles() : 0;
#endif
        float a = bias_f[a_oc];
        float b = bias_f[b_oc];

#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t bias_load_finished =
            profile_sample ? runtime_now_cycles() : 0;
#endif
        for (int64_t k = 0; k < K; k++) {
            const int64_t src_t = t + k - pad;
            if (src_t < 0 || src_t >= T) {
                continue;
            }
            const auto* source =
                reinterpret_cast<const float*>(
                    x_data + src_t * x->nb[1]);
            a += source[a_src_h] *
                 weight_f[
                     static_cast<size_t>(h) *
                         static_cast<size_t>(K) +
                     static_cast<size_t>(k)];
            b += source[b_src_h] *
                 weight_f[
                     (static_cast<size_t>(H) +
                      static_cast<size_t>(h)) *
                         static_cast<size_t>(K) +
                     static_cast<size_t>(k)];
        }

#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t mac_finished =
            profile_sample ? runtime_now_cycles() : 0;
#endif
        const float gate = 1.0f / (1.0f + std::exp(-b));
        auto* destination =
            reinterpret_cast<float*>(
                dst_data + t * dst->nb[1]);
        destination[h] = a * gate;
#if INFLECT_PROFILE_ACOUSTIC_OPS
        if (profile_sample) {
            const uint32_t gate_write_finished =
                runtime_now_cycles();
            bias_load_cycles += static_cast<uint32_t>(
                bias_load_finished - bias_load_started);
            mac_cycles += static_cast<uint32_t>(
                mac_finished - bias_load_finished);
            gate_write_cycles += static_cast<uint32_t>(
                gate_write_finished - mac_finished);
            sampled_outputs++;
        }
#endif
    }
#if INFLECT_PROFILE_ACOUSTIC_OPS
    acoustic_phase_profile_add(
        "depthwise.prepare",
        prepare_cycles,
        static_cast<uint64_t>(
            ggml_nelements(weight) +
            ggml_nelements(bias)));
    acoustic_phase_profile_add(
        "depthwise.bias_load",
        bias_load_cycles,
        sampled_outputs);
    acoustic_phase_profile_add(
        "depthwise.mac",
        mac_cycles,
        sampled_outputs);
    acoustic_phase_profile_add(
        "depthwise.gate_write",
        gate_write_cycles,
        sampled_outputs);
#endif
    acoustic_profile_add("depthwise_gated",
                         runtime_now_ms() - op_start_ms,
                         (uint64_t)(end - start),
                         (uint64_t)(end - start) * (uint64_t)K * 2ULL);
}

static ggml_tensor* depthwise_conv_gated(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    ggml_tensor* bias,
    int kernel_size,
    int padding,
    int dilation
) {
    (void)kernel_size;
    (void)padding;
    (void)dilation;
    return ggml_map_custom3(ctx, x, weight, bias, depthwise_gated_op, 1, nullptr);
}

// ═════════════════════════════════════════════════════════════════════════
// Conv Block
// ═════════════════════════════════════════════════════════════════════════

ggml_tensor* AcousticModel::build_conv_block(
    ggml_context* ctx,
    ggml_tensor* x,
    const ConvBlockWeights& w,
    int ff_mult,
    ggml_tensor* mask
) {
    (void)mask;
    (void)ff_mult;
    const int K = config_.kernel_size;

    // Conv branch: x + point(depthwise_gate(norm1(x)))
    ggml_tensor* residual = x;
    ggml_tensor* y = layer_norm(ctx, x, w.norm1_w, w.norm1_b);
    y = depthwise_conv_gated(ctx, y, w.depth_w, w.depth_b, K, K / 2, 1);
    y = linear(ctx, y, w.point_w, w.point_b);
    x = checked_add(ctx, residual, y, "conv block residual");

    // FFN branch: x + Linear(SiLU(Linear(norm2(x))))
    residual = x;
    y = layer_norm(ctx, x, w.norm2_w, w.norm2_b);
    y = linear(ctx, y, w.ff0_w, w.ff0_b);
    y = ggml_silu(ctx, y);
    y = linear(ctx, y, w.ff3_w, w.ff3_b);
    x = checked_add(ctx, residual, y, "ffn block residual");
    return x;
}

AcousticModel::AcousticModel(const AcousticConfig& config)
    : config_(config), weights_{} {}

AcousticModel::~AcousticModel() = default;

ggml_tensor* AcousticModel::layer_norm(ggml_context* ctx, ggml_tensor* x,
                                       ggml_tensor* w, ggml_tensor* b) {
    return inflect::layer_norm(ctx, x, w, b);
}

ggml_tensor* AcousticModel::linear(ggml_context* ctx, ggml_tensor* x,
                                   ggml_tensor* w, ggml_tensor* b) {
#if defined(INFLECT_LOW_MEMORY)
    if (ggml_tensor* packed = quantized_linear(
            ctx, x, w, b, quant_linear_ops_)) {
        return packed;
    }
#endif
    return inflect::linear(ctx, x, w, b);
}

ggml_tensor* AcousticModel::depthwise_conv_gated(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    ggml_tensor* bias,
    int kernel_size,
    int padding,
    int dilation
) {
    return inflect::depthwise_conv_gated(ctx, x, weight, bias, kernel_size, padding, dilation);
}

// ═════════════════════════════════════════════════════════════════════════
// Bidirectional GRU (direct buffer access for CPU backend)
// ═════════════════════════════════════════════════════════════════════════
struct GruOpData {
    ggml_tensor *w_ih, *w_hh, *b_ih, *b_hh;
    ggml_tensor *w_ih_r, *w_hh_r, *b_ih_r, *b_hh_r;
    int hidden_size;
};

static void gru_op(
    ggml_tensor* dst,
    const ggml_tensor* x,
    int ith,
    int nth,
    void* userdata
) {
    (void)nth;
    if (ith > 1) return;
    ACOUSTIC_PROFILE_START_MS(op_start_ms);
#if INFLECT_PROFILE_ACOUSTIC_OPS
    const uint32_t phase_total_started =
        runtime_now_cycles();
    uint64_t weight_unpack_cycles = 0;
    uint64_t state_alloc_cycles = 0;
    uint64_t bias_load_cycles = 0;
    uint64_t input_projection_cycles = 0;
    uint64_t input_dot_cycles = 0;
    uint64_t recurrent_projection_cycles = 0;
    uint64_t recurrent_dot_cycles = 0;
    uint64_t activation_cycles = 0;
    uint64_t state_write_cycles = 0;
    uint64_t weight_units = 0;
#endif

    const auto* d = static_cast<const GruOpData*>(userdata);
    const int input_size = x->ne[0];
    const int T = x->ne[1];
    const int hs = d->hidden_size;

    auto sigmoid = [](float v) {
        return 1.0f / (1.0f + std::exp(-v));
    };
    auto b1 = [](const ggml_tensor* t, int64_t i) {
        return tensor_get_f32(t, i, 0, 0);
    };

    auto run_direction = [&](bool reverse,
                             const ggml_tensor* w_ih, const ggml_tensor* w_hh,
                             const ggml_tensor* b_ih, const ggml_tensor* b_hh) {
        FloatScratch w_ih_f;
        FloatScratch w_hh_f;
        FloatScratch input_projection;
        FloatScratch recurrent_projection;
        FloatScratch recurrent_bias;
#if defined(INFLECT_LOW_MEMORY)
        AcousticQ4BatchDot recurrent_dot;
#if INFLECT_PROFILE_ACOUSTIC_OPS
        uint64_t recurrent_packed_total_cycles = 0;
        uint64_t recurrent_scratch_init_cycles = 0;
        uint64_t recurrent_cooperate_cycles = 0;
#endif
#endif
        bool packed_input_projection = false;
        bool packed_recurrent_projection = false;
#if defined(INFLECT_LOW_MEMORY)
#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t input_projection_started =
            runtime_now_cycles();
#endif
        if (w_ih->type == GGML_TYPE_Q4_0 &&
            x->type == GGML_TYPE_F32 &&
            x->nb[0] == sizeof(float) &&
            w_ih->ne[1] == 3 * hs) {
            input_projection.resize(
                static_cast<size_t>(T) *
                static_cast<size_t>(w_ih->ne[1]));
            packed_input_projection =
                acoustic_q4_linear_batch(
                    AcousticPackedProfileKind::GruInput,
                    w_ih,
                    b_ih,
                    static_cast<const float*>(x->data),
                    input_size,
                    static_cast<int>(
                        x->nb[1] / sizeof(float)),
                    T,
                    input_projection.data(),
                    static_cast<int>(w_ih->ne[1]),
                    1);
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        input_projection_cycles +=
            static_cast<uint32_t>(
                runtime_now_cycles() -
                input_projection_started);
#endif
        if (w_hh->type == GGML_TYPE_Q4_0 &&
            w_hh->ne[1] == 3 * hs) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
            const uint32_t recurrent_setup_started =
                runtime_now_cycles();
            const uint32_t recurrent_scratch_started =
                runtime_now_cycles();
#endif
            const bool recurrent_initialized =
                recurrent_dot.init(w_hh->ne[0], 1);
#if INFLECT_PROFILE_ACOUSTIC_OPS
            recurrent_scratch_init_cycles +=
                static_cast<uint32_t>(
                    runtime_now_cycles() -
                    recurrent_scratch_started);
#endif
            if (!recurrent_initialized ||
                !recurrent_dot.prepare_weights(w_hh)) {
                fprintf(stderr,
                        "[AcousticModel] packed GRU recurrent "
                        "preparation failed\n");
                std::abort();
            }
            recurrent_projection.resize(
                static_cast<size_t>(w_hh->ne[1]));
            tensor_row_to_f32(
                b_hh, 0, 0, recurrent_bias);
            packed_recurrent_projection = true;
#if INFLECT_PROFILE_ACOUSTIC_OPS
            recurrent_packed_total_cycles +=
                static_cast<uint32_t>(
                    runtime_now_cycles() -
                    recurrent_setup_started);
#endif
        }
#endif
#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t weight_unpack_started =
            runtime_now_cycles();
#endif
        if (!packed_input_projection) {
            tensor_rows_to_f32(w_ih, w_ih_f);
        }
        if (!packed_recurrent_projection) {
            tensor_rows_to_f32(w_hh, w_hh_f);
        }
#if INFLECT_PROFILE_ACOUSTIC_OPS
        weight_unpack_cycles += static_cast<uint32_t>(
            runtime_now_cycles() -
            weight_unpack_started);
        if (!packed_recurrent_projection) {
            weight_units += static_cast<uint64_t>(
                ggml_nelements(w_hh));
        }
        if (!packed_input_projection) {
            weight_units += static_cast<uint64_t>(
                ggml_nelements(w_ih));
        }
#endif
        const int w_ih_stride = (int)w_ih->ne[0];
        const int w_hh_stride = (int)w_hh->ne[0];
        const char* x_data = static_cast<const char*>(x->data);
        const int x_step = (int)(x->nb[0] / sizeof(float));

#if INFLECT_PROFILE_ACOUSTIC_OPS
        const uint32_t state_alloc_started =
            runtime_now_cycles();
#endif
        std::vector<float> h_prev(hs, 0.0f);
        std::vector<float> h_new(hs, 0.0f);
#if INFLECT_PROFILE_ACOUSTIC_OPS
        state_alloc_cycles += static_cast<uint32_t>(
            runtime_now_cycles() -
            state_alloc_started);
#endif

        for (int step = 0; step < T; step++) {
            const int t = reverse ? (T - 1 - step) : step;
            const float* x_t = reinterpret_cast<const float*>(x_data + t * x->nb[1]);
#if defined(INFLECT_LOW_MEMORY)
            if (packed_recurrent_projection) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
                const uint32_t recurrent_projection_started =
                    runtime_now_cycles();
                const uint32_t cooperate_started =
                    runtime_now_cycles();
#endif
                runtime_cooperate();
#if INFLECT_PROFILE_ACOUSTIC_OPS
                recurrent_cooperate_cycles +=
                    static_cast<uint32_t>(
                        runtime_now_cycles() -
                        cooperate_started);
#endif
                recurrent_dot.quantize_inputs(
                    h_prev.data(), hs, hs, 1);
                for (int output = 0;
                     output < w_hh->ne[1];
                     ++output) {
                    recurrent_dot.calculate_prepared(
                        output, 1);
                    recurrent_projection[output] =
                        recurrent_dot.result(0) +
                        recurrent_bias[output];
                }
#if INFLECT_PROFILE_ACOUSTIC_OPS
                const uint32_t recurrent_projection_elapsed =
                    static_cast<uint32_t>(
                        runtime_now_cycles() -
                        recurrent_projection_started);
                recurrent_projection_cycles +=
                    recurrent_projection_elapsed;
                recurrent_packed_total_cycles +=
                    recurrent_projection_elapsed;
#endif
            }
#endif
            for (int i = 0; i < hs; i++) {
#if INFLECT_PROFILE_ACOUSTIC_OPS
                const uint32_t bias_load_started =
                    runtime_now_cycles();
#endif
                float gi[3];
                if (!packed_input_projection) {
                    gi[0] = b1(b_ih, 0 * hs + i);
                    gi[1] = b1(b_ih, 1 * hs + i);
                    gi[2] = b1(b_ih, 2 * hs + i);
                }
                float gh[3];
                if (!packed_recurrent_projection) {
                    gh[0] = b1(b_hh, 0 * hs + i);
                    gh[1] = b1(b_hh, 1 * hs + i);
                    gh[2] = b1(b_hh, 2 * hs + i);
                }

#if INFLECT_PROFILE_ACOUSTIC_OPS
                const uint32_t bias_load_finished =
                    runtime_now_cycles();
#endif
                if (packed_input_projection) {
                    const float* projected =
                        input_projection.data() +
                        static_cast<size_t>(t) *
                            static_cast<size_t>(3 * hs);
                    gi[0] = projected[0 * hs + i];
                    gi[1] = projected[1 * hs + i];
                    gi[2] = projected[2 * hs + i];
                } else {
                    gi[0] += dot_f32_strided(
                        x_t, x_step,
                        w_ih_f.data() +
                            static_cast<size_t>(0 * hs + i) *
                                static_cast<size_t>(w_ih_stride),
                        1, input_size);
                    gi[1] += dot_f32_strided(
                        x_t, x_step,
                        w_ih_f.data() +
                            static_cast<size_t>(1 * hs + i) *
                                static_cast<size_t>(w_ih_stride),
                        1, input_size);
                    gi[2] += dot_f32_strided(
                        x_t, x_step,
                        w_ih_f.data() +
                            static_cast<size_t>(2 * hs + i) *
                                static_cast<size_t>(w_ih_stride),
                        1, input_size);
                }
#if INFLECT_PROFILE_ACOUSTIC_OPS
                const uint32_t input_dot_finished =
                    runtime_now_cycles();
#endif
                if (packed_recurrent_projection) {
                    gh[0] =
                        recurrent_projection[0 * hs + i];
                    gh[1] =
                        recurrent_projection[1 * hs + i];
                    gh[2] =
                        recurrent_projection[2 * hs + i];
                } else {
                    gh[0] += dot_f32(
                        h_prev.data(),
                        w_hh_f.data() +
                            static_cast<size_t>(0 * hs + i) *
                                static_cast<size_t>(w_hh_stride),
                        hs);
                    gh[1] += dot_f32(
                        h_prev.data(),
                        w_hh_f.data() +
                            static_cast<size_t>(1 * hs + i) *
                                static_cast<size_t>(w_hh_stride),
                        hs);
                    gh[2] += dot_f32(
                        h_prev.data(),
                        w_hh_f.data() +
                            static_cast<size_t>(2 * hs + i) *
                                static_cast<size_t>(w_hh_stride),
                        hs);
                }

#if INFLECT_PROFILE_ACOUSTIC_OPS
                const uint32_t recurrent_dot_finished =
                    runtime_now_cycles();
#endif
                // PyTorch GRU gate order is reset, update, new.
                const float r = sigmoid(gi[0] + gh[0]);
                const float z = sigmoid(gi[1] + gh[1]);
                const float n = std::tanh(gi[2] + r * gh[2]);
                h_new[i] = (1.0f - z) * n + z * h_prev[i];
#if INFLECT_PROFILE_ACOUSTIC_OPS
                const uint32_t activation_finished =
                    runtime_now_cycles();
                bias_load_cycles += static_cast<uint32_t>(
                    bias_load_finished -
                    bias_load_started);
                input_dot_cycles += static_cast<uint32_t>(
                    input_dot_finished -
                    bias_load_finished);
                recurrent_dot_cycles +=
                    static_cast<uint32_t>(
                        recurrent_dot_finished -
                        input_dot_finished);
                activation_cycles += static_cast<uint32_t>(
                    activation_finished -
                    recurrent_dot_finished);
#endif
            }

#if INFLECT_PROFILE_ACOUSTIC_OPS
            const uint32_t state_write_started =
                runtime_now_cycles();
#endif
            for (int i = 0; i < hs; i++) {
                h_prev[i] = h_new[i];
                tensor_set_f32(dst, h_new[i], (reverse ? hs : 0) + i, t, 0);
            }
#if INFLECT_PROFILE_ACOUSTIC_OPS
            state_write_cycles += static_cast<uint32_t>(
                runtime_now_cycles() -
                state_write_started);
#endif
        }
#if defined(INFLECT_LOW_MEMORY) && \
        INFLECT_PROFILE_ACOUSTIC_OPS
        if (packed_recurrent_projection) {
            AcousticPackedPhaseTimes completed =
                recurrent_dot.phases();
            completed.total_cycles +=
                recurrent_packed_total_cycles;
            completed.scratch_init_cycles +=
                recurrent_scratch_init_cycles;
            completed.cooperate_cycles +=
                recurrent_cooperate_cycles;
            completed.output_rows +=
                static_cast<uint64_t>(T) *
                static_cast<uint64_t>(w_hh->ne[1]);
            acoustic_packed_profile_add(
                AcousticPackedProfileKind::GruRecurrent,
                completed);
        }
#endif
    };

    if (ith == 0) {
        run_direction(false, d->w_ih, d->w_hh, d->b_ih, d->b_hh);
    } else {
        run_direction(true, d->w_ih_r, d->w_hh_r, d->b_ih_r, d->b_hh_r);
    }
#if INFLECT_PROFILE_ACOUSTIC_OPS
    const uint64_t phase_units =
        static_cast<uint64_t>(T) *
        static_cast<uint64_t>(hs);
    const uint64_t total_cycles =
        static_cast<uint32_t>(
            runtime_now_cycles() - phase_total_started);
    const uint64_t classified_cycles =
        weight_unpack_cycles +
        state_alloc_cycles +
        bias_load_cycles +
        input_projection_cycles +
        input_dot_cycles +
        recurrent_projection_cycles +
        recurrent_dot_cycles +
        activation_cycles +
        state_write_cycles;
    const uint64_t unclassified_cycles =
        total_cycles > classified_cycles
            ? total_cycles - classified_cycles
            : 0;
    acoustic_phase_profile_add(
        "gru.total", total_cycles, phase_units);
    acoustic_phase_profile_add(
        "gru.weight_unpack",
        weight_unpack_cycles,
        weight_units);
    acoustic_phase_profile_add(
        "gru.state_alloc",
        state_alloc_cycles,
        static_cast<uint64_t>(2 * hs));
    acoustic_phase_profile_add(
        "gru.bias_load",
        bias_load_cycles,
        phase_units);
    acoustic_phase_profile_add(
        "gru.input_projection",
        input_projection_cycles,
        static_cast<uint64_t>(T) *
            static_cast<uint64_t>(3 * hs));
    acoustic_phase_profile_add(
        "gru.input_dot",
        input_dot_cycles,
        phase_units);
    acoustic_phase_profile_add(
        "gru.recurrent_projection",
        recurrent_projection_cycles,
        static_cast<uint64_t>(T) *
            static_cast<uint64_t>(3 * hs));
    acoustic_phase_profile_add(
        "gru.recurrent_dot",
        recurrent_dot_cycles,
        phase_units);
    acoustic_phase_profile_add(
        "gru.activation",
        activation_cycles,
        phase_units);
    acoustic_phase_profile_add(
        "gru.state_write",
        state_write_cycles,
        phase_units);
    acoustic_phase_profile_add(
        "gru.unclassified",
        unclassified_cycles,
        phase_units);
#endif
    acoustic_profile_add("frame_gru",
                         runtime_now_ms() - op_start_ms,
                         (uint64_t)T * (uint64_t)hs,
                         (uint64_t)T * (uint64_t)hs *
                             (uint64_t)(3 * input_size + 3 * hs));
}

static ggml_tensor* bidirectional_gru(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* w_ih, ggml_tensor* w_hh,
    ggml_tensor* b_ih, ggml_tensor* b_hh,
    ggml_tensor* w_ih_r, ggml_tensor* w_hh_r,
    ggml_tensor* b_ih_r, ggml_tensor* b_hh_r,
    int hidden_size,
    GruOpData* data
) {
    *data = GruOpData{w_ih, w_hh, b_ih, b_hh, w_ih_r, w_hh_r, b_ih_r, b_hh_r, hidden_size};
    return ggml_map_custom1(ctx, x, gru_op, 2, data);
}

// ═════════════════════════════════════════════════════════════════════════
// Graph 1: Encoder
// ═════════════════════════════════════════════════════════════════════════

ggml_cgraph* AcousticModel::build_encoder_graph(
    ggml_context* gctx,
    ggml_tensor* phone_ids,  // [seq_len] int32
    ggml_tensor* tone_ids,   // [seq_len] int32
    ggml_tensor* lang_ids,   // [seq_len] int32
    ggml_tensor* speaker_ids // [1] int32
) {
    const int H = config_.hidden;

    // ── Embedding lookup ────────────────────────────────────────────
    // ggml_get_rows returns [embedding_width, n_ids].
    ggml_tensor* phone_emb = ggml_get_rows(gctx, weights_.phone_emb, phone_ids); // [H, T]
    phone_emb = ggml_cont(gctx, phone_emb);
    phone_emb = trim_ne0(gctx, phone_emb, H);

    ggml_tensor* tone_emb = ggml_get_rows(gctx, weights_.tone_emb, tone_ids);
    tone_emb = ggml_cont(gctx, tone_emb);
    tone_emb = trim_ne0(gctx, tone_emb, H);

    ggml_tensor* lang_emb = ggml_get_rows(gctx, weights_.lang_emb, lang_ids);
    lang_emb = ggml_cont(gctx, lang_emb);
    lang_emb = trim_ne0(gctx, lang_emb, H);

    // Speaker embedding
    ggml_tensor* spk_emb = ggml_get_rows(gctx, weights_.speaker_emb, speaker_ids); // [1, speaker_dim]
    spk_emb = ggml_cont(gctx, spk_emb);
    spk_emb = trim_ne0(gctx, spk_emb, config_.speaker_dim);
    spk_emb = linear(gctx, spk_emb, weights_.spk_proj_w, weights_.spk_proj_b);     // [H, 1]
    spk_emb = ggml_reshape_3d(gctx, spk_emb, H, 1, 1);                             // [H, 1, 1]

    // Sum embeddings (no sqrt(H) scaling in Inflect-Nano MicroFastSpeech)
    ggml_tensor* x = checked_add(gctx, checked_add(gctx, phone_emb, tone_emb, "phone + tone"), lang_emb, "embeddings + lang");
    x = checked_add(gctx, x, spk_emb, "embeddings + speaker");

    const bool capture_debug = debug_dump_enabled();
    ggml_tensor* embed_sum = capture_debug ? x : nullptr;
    if (embed_sum) {
        ggml_set_name(embed_sum, "embed_sum_internal");
    }

    // ── Encoder blocks ──────────────────────────────────────────────
    std::vector<ggml_tensor*> enc_block_outs;
    if (capture_debug) {
        enc_block_outs.reserve(config_.encoder_layers);
    }
    for (int i = 0; i < config_.encoder_layers; i++) {
        x = build_conv_block(gctx, x, weights_.enc_blocks[i],
                             config_.encoder_ff_mult, nullptr);
        if (capture_debug) {
            enc_block_outs.push_back(ggml_cpy(gctx, x, ggml_dup_tensor(gctx, x)));
        }
    }

    // ── Prediction heads ────────────────────────────────────────────
    // All heads read from the same encoded tensor
    // Duration
    ggml_tensor* dur = layer_norm(gctx, x, weights_.dur_norm_w, weights_.dur_norm_b);
    dur = linear(gctx, dur, weights_.dur_l1_w, weights_.dur_l1_b);
    dur = ggml_silu(gctx, dur);
    dur = linear(gctx, dur, weights_.dur_l2_w, weights_.dur_l2_b); // [1, T, 1]

    // Energy
    ggml_tensor* energy = layer_norm(gctx, x, weights_.energy_norm_w, weights_.energy_norm_b);
    energy = linear(gctx, energy, weights_.energy_l1_w, weights_.energy_l1_b);
    energy = ggml_silu(gctx, energy);
    energy = linear(gctx, energy, weights_.energy_l2_w, weights_.energy_l2_b); // [1, T, 1]

    // Bright
    ggml_tensor* bright = layer_norm(gctx, x, weights_.bright_norm_w, weights_.bright_norm_b);
    bright = linear(gctx, bright, weights_.bright_l1_w, weights_.bright_l1_b);
    bright = ggml_silu(gctx, bright);
    bright = linear(gctx, bright, weights_.bright_l2_w, weights_.bright_l2_b); // [1, T, 1]

    // Pitch
    ggml_tensor* pitch = layer_norm(gctx, x, weights_.pitch_norm_w, weights_.pitch_norm_b);
    pitch = linear(gctx, pitch, weights_.pitch_l1_w, weights_.pitch_l1_b);
    pitch = ggml_silu(gctx, pitch);
    pitch = linear(gctx, pitch, weights_.pitch_l2_w, weights_.pitch_l2_b); // [2, T, 1]

    // ── Output copies ───────────────────────────────────────────────
    ggml_tensor* embed_sum_out = capture_debug ? ggml_cpy(gctx, embed_sum, ggml_dup_tensor(gctx, embed_sum)) : nullptr;
    ggml_tensor* encoded_out = ggml_cpy(gctx, x, ggml_dup_tensor(gctx, x));
    ggml_tensor* dur_out     = ggml_cpy(gctx, dur, ggml_dup_tensor(gctx, dur));
    ggml_tensor* energy_out  = ggml_cpy(gctx, energy, ggml_dup_tensor(gctx, energy));
    ggml_tensor* bright_out  = ggml_cpy(gctx, bright, ggml_dup_tensor(gctx, bright));
    ggml_tensor* pitch_out   = ggml_cpy(gctx, pitch, ggml_dup_tensor(gctx, pitch));

    // ── Build graph ─────────────────────────────────────────────────
    ggml_cgraph* graph = ggml_new_graph_custom(gctx, 1024, false);

    if (capture_debug) {
        ggml_build_forward_expand(graph, embed_sum_out);
        for (ggml_tensor* block_out : enc_block_outs) {
            ggml_build_forward_expand(graph, block_out);
        }
    }
    ggml_build_forward_expand(graph, encoded_out);
    ggml_build_forward_expand(graph, dur_out);
    ggml_build_forward_expand(graph, energy_out);
    ggml_build_forward_expand(graph, bright_out);
    ggml_build_forward_expand(graph, pitch_out);
    // Store output tensor names for extraction
    if (capture_debug) {
        ggml_set_name(embed_sum_out, "embed_sum");
        for (int i = 0; i < (int)enc_block_outs.size(); i++) {
            ggml_set_name(enc_block_outs[i], ("enc_block_" + std::to_string(i)).c_str());
        }
    }
    ggml_set_name(encoded_out, "encoded");
    ggml_set_name(dur_out,     "log_durations");
    ggml_set_name(energy_out,  "energy");
    ggml_set_name(bright_out,  "bright");
    ggml_set_name(pitch_out,   "pitch");

    return graph;
}

// ═════════════════════════════════════════════════════════════════════════
// Graph 2: Decoder + Mel Head + Postnet
// ═════════════════════════════════════════════════════════════════════════

ggml_cgraph* AcousticModel::build_decoder_graph(
    ggml_context* gctx,
    ggml_tensor* features,  // [H, n_frames, 1]
    void* gru_op_data
) {
    const int H = config_.hidden;
    const int n_mels = config_.n_mels;
    (void)n_mels;

    ggml_tensor* x = features;

    // ── Decoder blocks ──────────────────────────────────────────────
    for (int i = 0; i < config_.decoder_layers; i++) {
        x = build_conv_block(gctx, x, weights_.dec_blocks[i],
                             config_.decoder_ff_mult, nullptr);
    }

    // ── Bidirectional GRU ───────────────────────────────────────────
    int gru_hidden = H / 2; // 84, concatenated to 168
    ggml_tensor* gru = inflect::bidirectional_gru(gctx, x,
        weights_.gru_w_ih, weights_.gru_w_hh,
        weights_.gru_b_ih, weights_.gru_b_hh,
        weights_.gru_w_ih_r, weights_.gru_w_hh_r,
        weights_.gru_b_ih_r, weights_.gru_b_hh_r,
        gru_hidden,
        static_cast<GruOpData*>(gru_op_data)); // [H, T, 1]
    x = checked_add(gctx, x, gru, "frame_gru residual");

    // ── Mel head ────────────────────────────────────────────────────
    x = layer_norm(gctx, x, weights_.mel_norm_w, weights_.mel_norm_b);
    x = linear(gctx, x, weights_.mel_l1_w, weights_.mel_l1_b);
    x = ggml_silu(gctx, x);
    x = linear(gctx, x, weights_.mel_l2_w, weights_.mel_l2_b); // [n_mels, T, 1]
    ggml_tensor* mel = x;

#if INFLECT_ACOUSTIC_SKIP_POSTNET
    x = mel;
#else
    // ── Postnet ─────────────────────────────────────────────────────
    // Postnet runs in GGML conv layout [T, C, B], then returns to [C, T, B].
    ggml_tensor* post = ggml_permute(gctx, mel, 1, 0, 2, 3);
    post = ggml_cont(gctx, post);
    post = quant_or_f16_conv1d(gctx, weights_.post0_w, post, 5, 1, 2, 1, quant_conv1d_ops_);
    post = checked_add(gctx, post, ggml_reshape_3d(gctx, weights_.post0_b, 1, H, 1), "postnet.0 bias");
    post = ggml_tanh(gctx, post);
    post = quant_or_f16_conv1d(gctx, weights_.post2_w, post, 5, 1, 2, 1, quant_conv1d_ops_);
    post = checked_add(gctx, post, ggml_reshape_3d(gctx, weights_.post2_b, 1, H, 1), "postnet.2 bias");
    post = ggml_tanh(gctx, post);
    post = quant_or_f16_conv1d(gctx, weights_.post4_w, post, 5, 1, 2, 1, quant_conv1d_ops_);
    post = checked_add(gctx, post, ggml_reshape_3d(gctx, weights_.post4_b, 1, n_mels, 1), "postnet.4 bias");
    post = ggml_permute(gctx, post, 1, 0, 2, 3);
    post = ggml_cont(gctx, post);
    // Scale postnet output and add to mel
    post = ggml_scale(gctx, post, config_.postnet_scale);
    x = checked_add(gctx, mel, post, "postnet residual");
#endif

    // ── Build graph ─────────────────────────────────────────────────
    ggml_set_name(x, "mel");
    ggml_cgraph* graph = ggml_new_graph_custom(gctx, 1024, false);
    ggml_build_forward_expand(graph, x);
    return graph;
}

// ═════════════════════════════════════════════════════════════════════════
// Graph Execution: Encoder
// ═════════════════════════════════════════════════════════════════════════

EncoderOutput AcousticModel::run_encoder(
    const std::vector<int32_t>& phone_ids,
    const std::vector<int32_t>& tone_ids,
    const std::vector<int32_t>& lang_ids,
    int speaker_id,
    ggml_backend_t backend
) {
    const int T = phone_ids.size();
    const int H = config_.hidden;
    EncoderOutput out;
    out.seq_len = T;
    out.hidden = H;

    // Create graph context
    size_t gctx_size = 512 * 1024;
    struct ggml_init_params gparams = {
        .mem_size   = gctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* gctx = ggml_init(gparams);

    // Create input tensors (these live in the backend buffer)
    ggml_tensor* phone_t  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_tensor* tone_t   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_tensor* lang_t   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_tensor* speaker_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);

    int32_t sid = speaker_id;

    // Build graph
    quant_linear_ops_.clear();
    quant_linear_ops_.reserve(32);
    ggml_cgraph* graph = build_encoder_graph(gctx, phone_t, tone_t, lang_t, speaker_t);

    // Allocate and compute
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!reserve_and_alloc_graph(allocr, graph, "encoder")) {
        ggml_gallocr_free(allocr);
        ggml_free(gctx);
        return out;
    }
    mem_trace_graph("encoder", gctx, allocr);

    // Set input tensors after allocation
    ggml_backend_tensor_set(phone_t, phone_ids.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(tone_t, tone_ids.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(lang_t, lang_ids.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(speaker_t, &sid, 0, sizeof(int32_t));

    // Compute
    acoustic_profile_reset();
    ACOUSTIC_PROFILE_START_MS(compute_started_ms);
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[AcousticModel] Encoder graph compute stopped: %s\n",
                ggml_status_to_string(status));
        ggml_gallocr_free(allocr);
        ggml_free(gctx);
        return {};
    }
    acoustic_profile_log(
        "encoder",
        runtime_now_ms() - compute_started_ms,
        graph);

    // Extract outputs
    auto to_host = [](ggml_tensor* t) -> std::vector<float> {
        std::vector<float> data(ggml_nelements(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        return data;
    };

    if (debug_dump_enabled()) {
        ggml_tensor* esum_t = ggml_get_tensor(gctx, "embed_sum");
        if (esum_t) {
            out.embed_sum = to_host(esum_t);
        }

        out.enc_blocks.clear();
        out.enc_blocks.reserve(config_.encoder_layers);
        for (int i = 0; i < config_.encoder_layers; i++) {
            ggml_tensor* block_t = ggml_get_tensor(gctx, ("enc_block_" + std::to_string(i)).c_str());
            if (block_t) {
                out.enc_blocks.push_back(to_host(block_t));
            }
        }
    }

    ggml_tensor* encoded_t = ggml_get_tensor(gctx, "encoded");
    ggml_tensor* dur_t     = ggml_get_tensor(gctx, "log_durations");
    ggml_tensor* energy_t  = ggml_get_tensor(gctx, "energy");
    ggml_tensor* bright_t  = ggml_get_tensor(gctx, "bright");
    ggml_tensor* pitch_t   = ggml_get_tensor(gctx, "pitch");
    if (!encoded_t || !dur_t || !energy_t || !bright_t || !pitch_t) {
        fprintf(stderr, "[AcousticModel] Failed to locate named encoder outputs\n");
        std::abort();
    }

    out.encoded       = to_host(encoded_t);
    out.log_durations = to_host(dur_t);
    out.energy        = to_host(energy_t);
    out.bright        = to_host(bright_t);
    out.pitch         = to_host(pitch_t);

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(gctx);

    return out;
}

// ═════════════════════════════════════════════════════════════════════════
// CPU Bridge: Length Regulation
//
// This expands phoneme-level features to frame-level based on predicted
// durations. Also adds prosody projections and absolute frame positional
// embeddings.
// ═════════════════════════════════════════════════════════════════════════

RegulatedFeatures AcousticModel::length_regulate(
    const EncoderOutput& enc,
    float length_scale,
    float pitch_scale,
    float energy_scale
) {
    ACOUSTIC_PROFILE_START_MS(bridge_started_ms);
    acoustic_profile_reset();
    const int H = config_.hidden;
    const int T = enc.seq_len;

    auto silu = [](float x) -> float {
        return x / (1.0f + std::exp(-x));
    };

    const float* dur_data    = enc.log_durations.data();
    const float* energy_pred = enc.energy.data();
    const float* bright_pred = enc.bright.data();
    const float* pitch_pred  = enc.pitch.data(); // GGML layout: component + token * 2

    FloatScratch unpacked_weight;
    auto linear_batch = [&](
        const float* input,
        int rows,
        int in_dim,
        const ggml_tensor* weight,
        const ggml_tensor* bias,
        float* output,
        int out_dim
    ) {
        ACOUSTIC_PROFILE_START_MS(started_ms);
        const uint64_t outputs =
            static_cast<uint64_t>(rows) *
            static_cast<uint64_t>(out_dim);
        const uint64_t macs =
            outputs * static_cast<uint64_t>(in_dim);
#if defined(INFLECT_LOW_MEMORY)
        constexpr uint64_t kMinimumPackedBridgeMacs = 32768;
        if (weight->type == GGML_TYPE_Q4_0 &&
            macs >= kMinimumPackedBridgeMacs &&
            acoustic_q4_linear_batch(
                AcousticPackedProfileKind::BridgeLinear,
                weight,
                bias,
                input,
                in_dim,
                in_dim,
                rows,
                output,
                out_dim,
                1)) {
            acoustic_profile_add(
                "bridge.linear",
                runtime_now_ms() - started_ms,
                outputs,
                macs);
            return;
        }
#endif
        tensor_rows_to_f32(weight, unpacked_weight);
        const int weight_stride =
            static_cast<int>(weight->ne[0]);
        if (weight_stride < in_dim ||
            weight->ne[1] * weight->ne[2] < out_dim) {
            fprintf(stderr,
                    "[AcousticModel] invalid bridge linear shape\n");
            std::abort();
        }
        for (int output_channel = 0;
             output_channel < out_dim;
             ++output_channel) {
            const float bias_value =
                bias != nullptr
                    ? tensor_get_f32(
                          bias, output_channel, 0, 0)
                    : 0.0f;
            const float* weight_row =
                unpacked_weight.data() +
                static_cast<size_t>(output_channel) *
                    static_cast<size_t>(weight_stride);
            for (int row = 0; row < rows; ++row) {
                output[
                    static_cast<size_t>(row) *
                        static_cast<size_t>(out_dim) +
                    static_cast<size_t>(output_channel)] =
                        bias_value +
                        dot_f32(
                            input +
                                static_cast<size_t>(row) *
                                    static_cast<size_t>(in_dim),
                            weight_row,
                            in_dim);
            }
        }
        acoustic_profile_add(
            "bridge.linear",
            runtime_now_ms() - started_ms,
            outputs,
            macs);
    };

    // ── Compute phoneme-level durations ─────────────────────────────
    // Apply expm1 with clamp, multiply by length_scale, round, clamp to at least 1
    std::vector<int32_t> durations(T);
    int total_frames = 0;
    for (int i = 0; i < T; i++) {
        float d = std::expm1(dur_data[i]);
        d = std::clamp(d, 0.0f, 80.0f);
        d *= length_scale;
        int di = (int)std::round(d);
        di = std::max(1, di);
        durations[i] = (int32_t)di;
        total_frames += di;
    }
    if (T <= 32) {
        fprintf(stderr, "[AcousticModel] durations total=%d values=", total_frames);
        for (int i = 0; i < T; i++) {
            fprintf(stderr, "%s%d", i == 0 ? "" : ",", (int)durations[i]);
        }
        fprintf(stderr, " log_durations=");
        for (int i = 0; i < T; i++) {
            fprintf(stderr, "%s%.3f", i == 0 ? "" : ",", dur_data[i]);
        }
        fprintf(stderr, "\n");
    }
    total_frames = std::min(total_frames, config_.max_frames);

    // Compute token-level conditioning in batches so each weight row is
    // unpacked once and reused across every token.
    std::vector<float> conditioned = enc.encoded;
    std::vector<float> token_projection(
        static_cast<size_t>(T) * static_cast<size_t>(H));
    std::vector<float> scalar_input(T);
    for (int t = 0; t < T; ++t) {
        scalar_input[t] = energy_pred[t] * energy_scale;
    }
    linear_batch(
        scalar_input.data(), T, 1,
        weights_.energy_proj_w, weights_.energy_proj_b,
        token_projection.data(), H);
    for (size_t i = 0; i < conditioned.size(); ++i) {
        conditioned[i] += token_projection[i];
    }
    for (int t = 0; t < T; ++t) {
        scalar_input[t] = bright_pred[t];
    }
    linear_batch(
        scalar_input.data(), T, 1,
        weights_.bright_proj_w, weights_.bright_proj_b,
        token_projection.data(), H);
    for (size_t i = 0; i < conditioned.size(); ++i) {
        conditioned[i] += token_projection[i];
    }

    // Local-context projections are token-level and can be evaluated as
    // two dense batches instead of repeating tensor reads per token.
    std::vector<float> context_input(
        static_cast<size_t>(T) * static_cast<size_t>(3 * H));
    for (int t = 0; t < T; ++t) {
        const int prev_t = std::max(0, t - 1);
        const int next_t = std::min(T - 1, t + 1);
        float* destination =
            context_input.data() +
            static_cast<size_t>(t) *
                static_cast<size_t>(3 * H);
        std::copy_n(
            conditioned.data() +
                static_cast<size_t>(prev_t) *
                    static_cast<size_t>(H),
            H,
            destination);
        std::copy_n(
            conditioned.data() +
                static_cast<size_t>(t) *
                    static_cast<size_t>(H),
            H,
            destination + H);
        std::copy_n(
            conditioned.data() +
                static_cast<size_t>(next_t) *
                    static_cast<size_t>(H),
            H,
            destination + 2 * H);
    }
    std::vector<float> context_hidden(
        static_cast<size_t>(T) * static_cast<size_t>(2 * H));
    std::vector<float> context_projection(
        static_cast<size_t>(T) * static_cast<size_t>(H));
    linear_batch(
        context_input.data(), T, 3 * H,
        weights_.lctx0_w, weights_.lctx0_b,
        context_hidden.data(), 2 * H);
    for (float& value : context_hidden) {
        value = silu(value);
    }
    linear_batch(
        context_hidden.data(), T, 2 * H,
        weights_.lctx2_w, weights_.lctx2_b,
        context_projection.data(), H);

    std::vector<float> pitch_input(
        static_cast<size_t>(T) * 2);
    for (int t = 0; t < T; ++t) {
        pitch_input[static_cast<size_t>(t) * 2] =
            pitch_pred[static_cast<size_t>(t) * 2] *
            pitch_scale;
        pitch_input[static_cast<size_t>(t) * 2 + 1] =
            std::clamp(
                pitch_pred[static_cast<size_t>(t) * 2 + 1],
                0.0f,
                1.0f);
    }
    std::vector<float> pitch_hidden(
        static_cast<size_t>(T) * static_cast<size_t>(H));
    std::vector<float> pitch_projection(
        static_cast<size_t>(T) * static_cast<size_t>(H));
    linear_batch(
        pitch_input.data(), T, 2,
        weights_.pitch_proj0_w, weights_.pitch_proj0_b,
        pitch_hidden.data(), H);
    for (float& value : pitch_hidden) {
        value = silu(value);
    }
    linear_batch(
        pitch_hidden.data(), T, H,
        weights_.pitch_proj2_w, weights_.pitch_proj2_b,
        pitch_projection.data(), H);

    // Build frame metadata once, then run both frame projections as
    // batched matrix operations.
    int n_frames = total_frames;
    std::vector<float> regulated(n_frames * H);
    std::vector<float> frame_metadata(
        static_cast<size_t>(n_frames) * 8);
    std::vector<int32_t> frame_tokens(n_frames);
    const int token_count = std::max(1, T);
    int f = 0;
    for (int t = 0; t < T && f < n_frames; t++) {
        const int dur = durations[t];
        for (int r = 0; r < dur && f < n_frames; r++, f++) {
            const float rel = dur > 1 ? (float)r / (float)(dur - 1) : 0.0f;
            float* metadata =
                frame_metadata.data() +
                static_cast<size_t>(f) * 8;
            metadata[0] = rel;
            metadata[1] = 1.0f - rel;
            metadata[2] =
                1.0f - std::fabs(rel * 2.0f - 1.0f);
            metadata[3] =
                std::sin(rel * (float)M_PI);
            metadata[4] =
                std::cos(rel * (float)M_PI);
            metadata[5] =
                static_cast<float>(t) /
                static_cast<float>(
                    std::max(1, token_count - 1));
            metadata[6] =
                std::log1p(static_cast<float>(dur)) / 6.0f;
            metadata[7] =
                static_cast<float>(dur) / 40.0f;
            frame_tokens[f] = t;
        }
    }
    std::vector<float> frame_hidden(
        static_cast<size_t>(n_frames) *
        static_cast<size_t>(H));
    linear_batch(
        frame_metadata.data(), n_frames, 8,
        weights_.frame_proj0_w, weights_.frame_proj0_b,
        frame_hidden.data(), H);
    for (float& value : frame_hidden) {
        value = silu(value);
    }
    linear_batch(
        frame_hidden.data(), n_frames, H,
        weights_.frame_proj2_w, weights_.frame_proj2_b,
        regulated.data(), H);

    for (int frame = 0; frame < n_frames; ++frame) {
        const int token = frame_tokens[frame];
        int bin =
            (frame * config_.abs_frame_bins) /
            std::max(1, config_.max_frames);
        bin = std::clamp(
            bin, 0, config_.abs_frame_bins - 1);
        const size_t frame_offset =
            static_cast<size_t>(frame) *
            static_cast<size_t>(H);
        const size_t token_offset =
            static_cast<size_t>(token) *
            static_cast<size_t>(H);
        for (int h = 0; h < H; ++h) {
            regulated[frame_offset + h] +=
                conditioned[token_offset + h] +
                context_projection[token_offset + h] +
                tensor_get_f32(
                    weights_.abs_frame_emb, h, bin, 0) +
                pitch_projection[token_offset + h];
        }
    }

    acoustic_profile_log(
        "bridge",
        runtime_now_ms() - bridge_started_ms,
        nullptr);
    return {regulated, durations, n_frames, H};
}

// ═════════════════════════════════════════════════════════════════════════
// Graph Execution: Decoder
// ═════════════════════════════════════════════════════════════════════════

std::vector<float> AcousticModel::run_decoder(
    const RegulatedFeatures& features,
    ggml_backend_t backend
) {
    const int H = config_.hidden;

    // Create graph context + input tensor
    size_t gctx_size = 512 * 1024;
    struct ggml_init_params gparams = {
        .mem_size   = gctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* gctx = ggml_init(gparams);
    mem_trace_rss("decoder ctx init");

    // Create FEATURES input tensor
    ggml_tensor* features_t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, H, features.n_frames, 1);
    ggml_set_input(features_t);
    mem_trace_rss("decoder input tensor");

    // Build graph
    GruOpData gru_data{};
    quant_linear_ops_.clear();
    quant_linear_ops_.reserve(32);
    quant_conv1d_ops_.clear();
    quant_conv1d_ops_.reserve(3);
    ggml_cgraph* graph = build_decoder_graph(gctx, features_t, &gru_data);
    mem_trace_rss("decoder graph built");

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!reserve_and_alloc_graph(allocr, graph, "decoder")) {
        ggml_gallocr_free(allocr);
        ggml_free(gctx);
        return {};
    }
    mem_trace_graph("decoder", gctx, allocr);
    mem_trace_rss("decoder allocated");

    // Set input
    ggml_backend_tensor_set(features_t, features.features.data(), 0, features.features.size() * sizeof(float));
#if defined(INFLECT_LOW_MEMORY)
    mem_release_to_os();
#endif
    mem_trace_rss("decoder input copied");

    // Compute
    acoustic_profile_reset();
    ACOUSTIC_PROFILE_START_MS(compute_start_ms);
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[AcousticModel] Decoder graph compute stopped: %s\n",
                ggml_status_to_string(status));
        ggml_gallocr_free(allocr);
        ggml_free(gctx);
        return {};
    }
    mem_trace_rss("decoder computed");
    acoustic_profile_log(
        "decoder",
        runtime_now_ms() - compute_start_ms,
        graph);

    // Extract mel
    ggml_tensor* mel_t = ggml_get_tensor(gctx, "mel");
    if (!mel_t) {
        fprintf(stderr, "[AcousticModel] mel tensor not found\n");
        ggml_gallocr_free(allocr);
        ggml_free(gctx);
        return {};
    }

    std::vector<float> mel(ggml_nelements(mel_t));
    ggml_backend_tensor_get(mel_t, mel.data(), 0, ggml_nbytes(mel_t));
    mem_trace_rss("decoder mel copied");

    ggml_gallocr_free(allocr);
#if defined(INFLECT_LOW_MEMORY)
    mem_release_to_os();
#endif
    mem_trace_rss("decoder allocator freed");
    ggml_free(gctx);
#if defined(INFLECT_LOW_MEMORY)
    mem_release_to_os();
#endif
    mem_trace_rss("decoder context freed");

    return mel;
}

// ═════════════════════════════════════════════════════════════════════════
// Weights Loading
// ═════════════════════════════════════════════════════════════════════════

void AcousticModel::load_conv_block(ModelLoader& loader, ConvBlockWeights& blk,
                                     const std::string& prefix) {
    blk.norm1_w  = loader.get_tensor(prefix + ".norm1.weight");
    blk.norm1_b  = loader.get_tensor(prefix + ".norm1.bias");
    blk.depth_w  = loader.get_tensor(prefix + ".depth.weight");
    blk.depth_b  = loader.get_tensor(prefix + ".depth.bias");
    blk.point_w  = loader.get_tensor(prefix + ".point.weight");
    blk.point_b  = loader.get_tensor(prefix + ".point.bias");
    blk.norm2_w  = loader.get_tensor(prefix + ".norm2.weight");
    blk.norm2_b  = loader.get_tensor(prefix + ".norm2.bias");
    blk.ff0_w    = loader.get_tensor(prefix + ".ff.0.weight");
    blk.ff0_b    = loader.get_tensor(prefix + ".ff.0.bias");
    blk.ff3_w    = loader.get_tensor(prefix + ".ff.3.weight");
    blk.ff3_b    = loader.get_tensor(prefix + ".ff.3.bias");
}

bool AcousticModel::load_encoder(ModelLoader& loader) {
    weights_.enc_blocks.clear();
    weights_.dec_blocks.clear();

    // Embeddings
    weights_.phone_emb   = loader.get_tensor("phone.weight");
    weights_.tone_emb    = loader.get_tensor("tone.weight");
    weights_.lang_emb    = loader.get_tensor("lang.weight");
    weights_.speaker_emb = loader.get_tensor("speaker.weight");
    weights_.spk_proj_w  = loader.get_tensor("speaker_proj.weight");
    weights_.spk_proj_b  = loader.get_tensor("speaker_proj.bias");

    // Encoder blocks
    for (int i = 0; i < config_.encoder_layers; i++) {
        ConvBlockWeights blk;
        load_conv_block(loader, blk, "encoder." + std::to_string(i));
        weights_.enc_blocks.push_back(blk);
    }

    // Duration head
    weights_.dur_norm_w = loader.get_tensor("duration_head.0.weight");
    weights_.dur_norm_b = loader.get_tensor("duration_head.0.bias");
    weights_.dur_l1_w   = loader.get_tensor("duration_head.1.weight");
    weights_.dur_l1_b   = loader.get_tensor("duration_head.1.bias");
    weights_.dur_l2_w   = loader.get_tensor("duration_head.3.weight");
    weights_.dur_l2_b   = loader.get_tensor("duration_head.3.bias");

    // Energy head
    weights_.energy_norm_w = loader.get_tensor("energy_head.0.weight");
    weights_.energy_norm_b = loader.get_tensor("energy_head.0.bias");
    weights_.energy_l1_w   = loader.get_tensor("energy_head.1.weight");
    weights_.energy_l1_b   = loader.get_tensor("energy_head.1.bias");
    weights_.energy_l2_w   = loader.get_tensor("energy_head.3.weight");
    weights_.energy_l2_b   = loader.get_tensor("energy_head.3.bias");

    // Bright head
    weights_.bright_norm_w = loader.get_tensor("bright_head.0.weight");
    weights_.bright_norm_b = loader.get_tensor("bright_head.0.bias");
    weights_.bright_l1_w   = loader.get_tensor("bright_head.1.weight");
    weights_.bright_l1_b   = loader.get_tensor("bright_head.1.bias");
    weights_.bright_l2_w   = loader.get_tensor("bright_head.3.weight");
    weights_.bright_l2_b   = loader.get_tensor("bright_head.3.bias");

    // Pitch head
    weights_.pitch_norm_w = loader.get_tensor("pitch_head.0.weight");
    weights_.pitch_norm_b = loader.get_tensor("pitch_head.0.bias");
    weights_.pitch_l1_w   = loader.get_tensor("pitch_head.1.weight");
    weights_.pitch_l1_b   = loader.get_tensor("pitch_head.1.bias");
    weights_.pitch_l2_w   = loader.get_tensor("pitch_head.3.weight");
    weights_.pitch_l2_b   = loader.get_tensor("pitch_head.3.bias");

    // Prosody projections
    weights_.energy_proj_w  = loader.get_tensor("energy_proj.weight");
    weights_.energy_proj_b  = loader.get_tensor("energy_proj.bias");
    weights_.bright_proj_w  = loader.get_tensor("bright_proj.weight");
    weights_.bright_proj_b  = loader.get_tensor("bright_proj.bias");
    weights_.pitch_proj0_w  = loader.get_tensor("pitch_proj.0.weight");
    weights_.pitch_proj0_b  = loader.get_tensor("pitch_proj.0.bias");
    weights_.pitch_proj2_w  = loader.get_tensor("pitch_proj.2.weight");
    weights_.pitch_proj2_b  = loader.get_tensor("pitch_proj.2.bias");

    // Frame-level features
    weights_.abs_frame_emb  = loader.get_tensor("abs_frame.weight");
    weights_.frame_proj0_w  = loader.get_tensor("frame_proj.0.weight");
    weights_.frame_proj0_b  = loader.get_tensor("frame_proj.0.bias");
    weights_.frame_proj2_w  = loader.get_tensor("frame_proj.2.weight");
    weights_.frame_proj2_b  = loader.get_tensor("frame_proj.2.bias");

    // Local context
    weights_.lctx0_w = loader.get_tensor("local_ctx.0.weight");
    weights_.lctx0_b = loader.get_tensor("local_ctx.0.bias");
    weights_.lctx2_w = loader.get_tensor("local_ctx.2.weight");
    weights_.lctx2_b = loader.get_tensor("local_ctx.2.bias");

    wctx_ = loader.ctx();
    return true;
}

bool AcousticModel::load_decoder(ModelLoader& loader) {
    weights_.dec_blocks.clear();

    for (int i = 0; i < config_.decoder_layers; i++) {
        ConvBlockWeights blk;
        load_conv_block(loader, blk, "decoder." + std::to_string(i));
        weights_.dec_blocks.push_back(blk);
    }

    weights_.gru_w_ih   = loader.get_tensor("frame_gru.weight_ih_l0");
    weights_.gru_w_hh   = loader.get_tensor("frame_gru.weight_hh_l0");
    weights_.gru_b_ih   = loader.get_tensor("frame_gru.bias_ih_l0");
    weights_.gru_b_hh   = loader.get_tensor("frame_gru.bias_hh_l0");
    weights_.gru_w_ih_r = loader.get_tensor("frame_gru.weight_ih_l0_reverse");
    weights_.gru_w_hh_r = loader.get_tensor("frame_gru.weight_hh_l0_reverse");
    weights_.gru_b_ih_r = loader.get_tensor("frame_gru.bias_ih_l0_reverse");
    weights_.gru_b_hh_r = loader.get_tensor("frame_gru.bias_hh_l0_reverse");

    weights_.mel_norm_w = loader.get_tensor("mel_head.0.weight");
    weights_.mel_norm_b = loader.get_tensor("mel_head.0.bias");
    weights_.mel_l1_w   = loader.get_tensor("mel_head.1.weight");
    weights_.mel_l1_b   = loader.get_tensor("mel_head.1.bias");
    weights_.mel_l2_w   = loader.get_tensor("mel_head.3.weight");
    weights_.mel_l2_b   = loader.get_tensor("mel_head.3.bias");

#if !INFLECT_ACOUSTIC_SKIP_POSTNET
    weights_.post0_w = loader.get_tensor("postnet.0.weight");
    weights_.post0_b = loader.get_tensor("postnet.0.bias");
    weights_.post2_w = loader.get_tensor("postnet.2.weight");
    weights_.post2_b = loader.get_tensor("postnet.2.bias");
    weights_.post4_w = loader.get_tensor("postnet.4.weight");
    weights_.post4_b = loader.get_tensor("postnet.4.bias");
#endif

    wctx_ = loader.ctx();
    return true;
}

bool AcousticModel::load(ModelLoader& loader) {
    return load_encoder(loader) && load_decoder(loader);
}

} // namespace inflect
