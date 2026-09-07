#include "sano_piper_model.h"

#include "memory_trace.h"
#include "vocoder_model.h"

#include "../ggml/include/ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace inflect {
namespace {

#ifndef INFLECT_SANO_MAX_FRAMES
#if defined(INFLECT_LOW_MEMORY)
#define INFLECT_SANO_MAX_FRAMES 128
#else
#define INFLECT_SANO_MAX_FRAMES 0
#endif
#endif

struct EmbeddingCache {
    const ggml_tensor* tensor = nullptr;
    std::vector<uint8_t> bytes;
    const char* base = nullptr;

    bool load(const ggml_tensor* value, int hidden, int vocab) {
        tensor = value;
        if (!tensor || ggml_is_quantized(tensor->type) ||
            (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) ||
            tensor->ne[0] < hidden || tensor->ne[1] < vocab) {
            return false;
        }
        if (tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer)) {
            base = static_cast<const char*>(tensor->data);
        } else {
            bytes.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
            base = reinterpret_cast<const char*>(bytes.data());
        }
        return true;
    }

    float at(int row, int column) const {
        const char* ptr = base + static_cast<size_t>(row) * tensor->nb[1] +
                          static_cast<size_t>(column) * tensor->nb[0];
        if (tensor->type == GGML_TYPE_F32) {
            return *reinterpret_cast<const float*>(ptr);
        }
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(ptr));
    }
};

bool valid_bias(const ggml_tensor* bias, int out_channels) {
    return bias != nullptr && bias->ne[0] == out_channels &&
           ggml_nelements(bias) >= out_channels;
}

bool valid_conv(
    const ggml_tensor* weight,
    const ggml_tensor* bias,
    int in_channels,
    int out_channels,
    int kernel
) {
    if (!weight || !valid_bias(bias, out_channels)) return false;
    if (ggml_is_quantized(weight->type)) {
        return weight->ne[0] >= static_cast<int64_t>(in_channels) * kernel &&
               weight->ne[1] == out_channels;
    }
    return weight->ne[0] == kernel && weight->ne[1] == in_channels &&
           weight->ne[2] == out_channels;
}

bool valid_scalar(const ggml_tensor* value) {
    return value != nullptr && ggml_nelements(value) == 1;
}

ggml_tensor* scalar_f32(ggml_context* ctx, ggml_tensor* value) {
    return value->type == GGML_TYPE_F32
               ? value
               : ggml_cast(ctx, value, GGML_TYPE_F32);
}

void linspace01(float* output, int count) {
    if (count <= 0) return;
    if (count == 1) {
        output[0] = 0.0f;
        return;
    }
    const float step = 1.0f / static_cast<float>(count - 1);
    const int half = count / 2;
    for (int index = 0; index < half; ++index) {
        output[index] = step * static_cast<float>(index);
    }
    for (int index = half; index < count; ++index) {
        output[index] = std::fma(
            -step, static_cast<float>(count - 1 - index), 1.0f);
    }
}

ggml_context* new_front_context(size_t bytes = 192 * 1024) {
    struct ggml_init_params params = {
        .mem_size = bytes,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    return ggml_init(params);
}

bool allocate_graph(
    ggml_context* ctx,
    ggml_cgraph* graph,
    ggml_backend_t backend,
    ggml_gallocr_t& allocator,
    const char* label
) {
    allocator =
        ggml_gallocr_new(runtime_weight_buffer_type());
    if (!allocator || !ggml_gallocr_reserve(allocator, graph) ||
        !ggml_gallocr_alloc_graph(allocator, graph)) {
        std::fprintf(stderr, "[SanoPiper] failed to allocate %s graph\n", label);
        if (allocator) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
        return false;
    }
    mem_trace_graph(label, ctx, allocator);
    std::fprintf(stderr, "[SanoGraph] %s graph_bytes=%zu\n", label,
                 ggml_gallocr_get_buffer_size(allocator, 0));
    (void)backend;
    return true;
}

bool set_channel(
    ggml_tensor* input,
    int channel,
    const float* values,
    int frames
) {
    if (!input || input->type != GGML_TYPE_F32 || channel < 0 ||
        channel >= input->ne[1] || input->ne[0] != frames) {
        return false;
    }
    ggml_backend_tensor_set(
        input,
        values,
        static_cast<size_t>(channel) * static_cast<size_t>(frames) * sizeof(float),
        static_cast<size_t>(frames) * sizeof(float));
    return true;
}

ggml_tensor* build_residual_stack(
    ggml_context* ctx,
    ModelLoader& loader,
    ggml_tensor* input,
    const std::string& block_prefix,
    int hidden,
    int depth,
    int kernel,
    std::vector<VocoderQuantConv1dOpData>& conv_ops,
    const char* profile_prefix
) {
    ggml_tensor* value = input;
    for (int block = 0; block < depth; ++block) {
        const std::string root = block_prefix + std::to_string(block) + ".";
        ggml_tensor* scale = loader.get_tensor(root + "scale");
        ggml_tensor* conv1_w = loader.get_tensor(root + "net.0.weight");
        ggml_tensor* conv1_b = loader.get_tensor(root + "net.0.bias");
        ggml_tensor* conv2_w = loader.get_tensor(root + "net.2.weight");
        ggml_tensor* conv2_b = loader.get_tensor(root + "net.2.bias");
        if (!valid_scalar(scale) ||
            !valid_conv(conv1_w, conv1_b, hidden, hidden, kernel) ||
            !valid_conv(conv2_w, conv2_b, hidden, hidden, kernel)) {
            std::fprintf(stderr,
                         "[SanoPiper] invalid residual tensor group %s\n",
                         root.c_str());
            return nullptr;
        }
        ggml_tensor* branch = packed_conv::conv1d(
            ctx, conv1_w, conv1_b, value, kernel, 1, kernel / 2, 1,
            conv_ops, profile_prefix);
        branch = ggml_silu(ctx, branch);
        branch = packed_conv::conv1d(
            ctx, conv2_w, conv2_b, branch, kernel, 1, kernel / 2, 1,
            conv_ops, profile_prefix);
        branch = ggml_mul(ctx, branch, scalar_f32(ctx, scale));
        value = ggml_add(ctx, value, branch);
    }
    return value;
}

ggml_tensor* build_front_graph(
    ggml_context* ctx,
    ModelLoader& loader,
    ggml_tensor* input,
    const std::string& input_projection,
    const std::string& block_prefix,
    const std::string& output_projection,
    int input_channels,
    int hidden,
    int depth,
    int kernel,
    int output_channels,
    std::vector<VocoderQuantConv1dOpData>& conv_ops,
    const char* profile_prefix
) {
    ggml_tensor* input_w = loader.get_tensor(input_projection + ".weight");
    ggml_tensor* input_b = loader.get_tensor(input_projection + ".bias");
    if (!valid_conv(input_w, input_b, input_channels, hidden, 1)) return nullptr;

    ggml_tensor* value = packed_conv::conv1d(
        ctx, input_w, input_b, input, 1, 1, 0, 1, conv_ops, profile_prefix);
    value = build_residual_stack(
        ctx, loader, value, block_prefix, hidden, depth, kernel,
        conv_ops, profile_prefix);
    if (!value) return nullptr;

    if (!output_projection.empty()) {
        ggml_tensor* output_w = loader.get_tensor(output_projection + ".weight");
        ggml_tensor* output_b = loader.get_tensor(output_projection + ".bias");
        if (!valid_conv(output_w, output_b, hidden, output_channels, 1)) {
            return nullptr;
        }
        value = packed_conv::conv1d(
            ctx, output_w, output_b, value, 1, 1, 0, 1,
            conv_ops, profile_prefix);
    }
    return value;
}

std::vector<float> read_output(
    ggml_tensor* output,
    ggml_backend_t backend,
    ggml_cgraph* graph,
    ggml_gallocr_t allocator,
    const char* label
) {
    std::vector<float> result;
    const uint32_t started = runtime_now_ms();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[SanoPiper] %s graph failed: %s\n",
                     label, ggml_status_to_string(status));
        ggml_gallocr_free(allocator);
        return result;
    }
    if (output->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[SanoPiper] %s output is not F32\n", label);
        ggml_gallocr_free(allocator);
        return result;
    }
#if defined(__cpp_exceptions)
    try {
#endif
        result.resize(static_cast<size_t>(ggml_nelements(output)));
#if defined(__cpp_exceptions)
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "[SanoMemory] %s output allocation failed\n", label);
        ggml_gallocr_free(allocator);
        return {};
    }
#endif
    ggml_backend_tensor_get(
        output, result.data(), 0, result.size() * sizeof(float));
    ggml_gallocr_free(allocator);
    std::fprintf(stderr, "[SanoStage] %s values=%zu compute_copy_ms=%u\n",
                 label, result.size(),
                 static_cast<unsigned>(runtime_now_ms() - started));
#if defined(INFLECT_LOW_MEMORY)
    mem_release_to_os();
#endif
    return result;
}

std::vector<float> run_duration_logits(
    ModelLoader& loader,
    const SanoPiperConfig& config,
    const std::vector<int32_t>& tokens,
    ggml_backend_t backend
) {
    const int count = static_cast<int>(tokens.size());
    const int hidden = config.duration_hidden;
    packed_conv::Q4WeightCache weight_cache;
    ggml_tensor* embedding = loader.get_tensor("duration.embedding.weight");
    EmbeddingCache cache;
    if (count <= 0 || !cache.load(embedding, hidden, config.duration_vocab)) {
        return {};
    }

    ggml_context* ctx = new_front_context();
    if (!ctx) return {};
    std::vector<VocoderQuantConv1dOpData> conv_ops;
    conv_ops.reserve(static_cast<size_t>(config.duration_depth) * 2 + 2);
    ggml_tensor* input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, count, hidden + 3, 1);
    ggml_set_input(input);
    ggml_tensor* output = build_front_graph(
        ctx, loader, input,
        "duration.input_proj", "duration.blocks.", "duration.output",
        hidden + 3, hidden, config.duration_depth, config.duration_kernel, 1,
        conv_ops, "sano.duration");
    if (!output) {
        ggml_free(ctx);
        return {};
    }
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = nullptr;
    if (!allocate_graph(ctx, graph, backend, allocator, "sano.duration")) {
        ggml_free(ctx);
        return {};
    }

    std::vector<float> row(static_cast<size_t>(count));
    for (int channel = 0; channel < hidden; ++channel) {
        for (int index = 0; index < count; ++index) {
            const int id = config.clamp_id(tokens[index], config.duration_vocab);
            row[index] = cache.at(id, channel);
        }
        set_channel(input, channel, row.data(), count);
    }
    linspace01(row.data(), count);
    set_channel(input, hidden, row.data(), count);
    const float hint =
        ::log1pf(static_cast<float>(count)) /
        static_cast<float>(::log1p(static_cast<double>(config.duration_max_tokens)));
    std::fill(row.begin(), row.end(), hint);
    set_channel(input, hidden + 1, row.data(), count);
    std::fill(row.begin(), row.end(), 1.0f);
    set_channel(input, hidden + 2, row.data(), count);

    std::vector<float> result =
        read_output(output, backend, graph, allocator, "sano.duration");
    ggml_free(ctx);
    return result;
}

std::vector<float> run_token_context(
    ModelLoader& loader,
    const SanoPiperConfig& config,
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& durations,
    ggml_backend_t backend
) {
    const int count = static_cast<int>(tokens.size());
    const int hidden = config.acoustic_hidden;
    packed_conv::Q4WeightCache weight_cache;
    ggml_tensor* embedding = loader.get_tensor("acoustic.embedding.weight");
    EmbeddingCache cache;
    if (count <= 0 || durations.size() != tokens.size() ||
        !cache.load(embedding, hidden, config.acoustic_vocab)) {
        return {};
    }

    ggml_context* ctx = new_front_context();
    if (!ctx) return {};
    std::vector<VocoderQuantConv1dOpData> conv_ops;
    conv_ops.reserve(static_cast<size_t>(config.acoustic_token_depth) * 2 + 1);
    ggml_tensor* input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, count, hidden + 2, 1);
    ggml_set_input(input);
    ggml_tensor* output = build_front_graph(
        ctx, loader, input,
        "acoustic.token_input_proj", "acoustic.token_blocks.", "",
        hidden + 2, hidden, config.acoustic_token_depth, config.acoustic_kernel,
        hidden, conv_ops, "sano.acoustic.token");
    if (!output) {
        ggml_free(ctx);
        return {};
    }
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = nullptr;
    if (!allocate_graph(ctx, graph, backend, allocator, "sano.acoustic.token")) {
        ggml_free(ctx);
        return {};
    }

    std::vector<float> row(static_cast<size_t>(count));
    for (int channel = 0; channel < hidden; ++channel) {
        for (int index = 0; index < count; ++index) {
            const int id = config.clamp_id(tokens[index], config.acoustic_vocab);
            row[index] = cache.at(id, channel);
        }
        set_channel(input, channel, row.data(), count);
    }
    linspace01(row.data(), count);
    set_channel(input, hidden, row.data(), count);
    int32_t max_duration = 1;
    for (int32_t duration : durations) {
        max_duration = std::max(max_duration, duration);
    }
    const float log_max = ::log1pf(static_cast<float>(max_duration));
    for (int index = 0; index < count; ++index) {
        row[index] = ::log1pf(static_cast<float>(durations[index])) / log_max;
    }
    set_channel(input, hidden + 1, row.data(), count);

    std::vector<float> result =
        read_output(output, backend, graph, allocator, "sano.acoustic.token");
    ggml_free(ctx);
    return result;
}

std::vector<float> run_frame_stage(
    ModelLoader& loader,
    const SanoPiperConfig& config,
    std::vector<float>& token_context,
    const std::vector<int32_t>& durations,
    int frames,
    ggml_backend_t backend
) {
    const int token_count = static_cast<int>(durations.size());
    packed_conv::Q4WeightCache weight_cache;
    const int hidden = config.acoustic_hidden;
    if (frames <= 0 || token_count <= 0 ||
        token_context.size() != static_cast<size_t>(hidden) * token_count) {
        return {};
    }

    ggml_context* ctx = new_front_context(256 * 1024);
    if (!ctx) return {};
    std::vector<VocoderQuantConv1dOpData> conv_ops;
    conv_ops.reserve(static_cast<size_t>(config.acoustic_depth) * 2 + 2);
    ggml_tensor* input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, frames, hidden + 3, 1);
    ggml_set_input(input);
    ggml_tensor* output = build_front_graph(
        ctx, loader, input,
        "acoustic.frame_input_proj", "acoustic.frame_blocks.", "acoustic.output",
        hidden + 3, hidden, config.acoustic_depth, config.acoustic_kernel,
        config.acoustic_out_channels, conv_ops, "sano.acoustic.frame");
    if (!output) {
        ggml_free(ctx);
        return {};
    }
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 768, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = nullptr;
    if (!allocate_graph(ctx, graph, backend, allocator, "sano.acoustic.frame")) {
        ggml_free(ctx);
        return {};
    }

    std::vector<float> row(static_cast<size_t>(frames));
    for (int channel = 0; channel < hidden; ++channel) {
        int position = 0;
        const float* context_row =
            token_context.data() + static_cast<size_t>(channel) * token_count;
        for (int token = 0; token < token_count; ++token) {
            std::fill_n(row.data() + position, durations[token], context_row[token]);
            position += durations[token];
        }
        set_channel(input, channel, row.data(), frames);
    }
    linspace01(row.data(), frames);
    set_channel(input, hidden, row.data(), frames);

    int position = 0;
    const int token_denominator = token_count > 1 ? token_count - 1 : 1;
    for (int token = 0; token < token_count; ++token) {
        const float token_position = static_cast<float>(
            static_cast<double>(token) / static_cast<double>(token_denominator));
        std::fill_n(row.data() + position, durations[token], token_position);
        position += durations[token];
    }
    set_channel(input, hidden + 1, row.data(), frames);

    position = 0;
    for (int token = 0; token < token_count; ++token) {
        const int duration = durations[token];
        if (duration == 1) {
            row[position++] = 0.0f;
            continue;
        }
        for (int index = 0; index < duration; ++index) {
            row[position++] = static_cast<float>(
                static_cast<double>(index) /
                static_cast<double>(duration - 1));
        }
    }
    set_channel(input, hidden + 2, row.data(), frames);

    // The expanded graph input now owns everything needed from token context.
    std::vector<float>().swap(token_context);
#if defined(INFLECT_LOW_MEMORY)
    mem_release_to_os();
#endif
    std::vector<float> result =
        read_output(output, backend, graph, allocator, "sano.acoustic.frame");
    ggml_free(ctx);
    return result;
}

} // namespace

bool SanoPiperModel::configure(ModelLoader& loader) {
    if (!config_.load(loader)) return false;
    decoder_.configure(config_);
    return true;
}

std::vector<int32_t> SanoPiperModel::predict_durations(
    ModelLoader& loader,
    const std::vector<int32_t>& token_ids,
    float speaking_rate,
    ggml_backend_t backend
) const {
    if (token_ids.empty() ||
        token_ids.size() > static_cast<size_t>(config_.duration_max_tokens) ||
        !(speaking_rate > 0.0f) || !std::isfinite(speaking_rate)) {
        return {};
    }
    const std::vector<float> logits =
        run_duration_logits(loader, config_, token_ids, backend);
    if (logits.size() != token_ids.size()) return {};

    const float scale = config_.duration_length_scale * speaking_rate;
    std::vector<int32_t> durations(logits.size());
    int64_t total_frames = 0;
    for (size_t index = 0; index < logits.size(); ++index) {
        if (!std::isfinite(logits[index])) return {};
        float value = ::expf(logits[index]);
        if (value < 1.0f) value = 1.0f;
        // Clamp before converting to int: expf can overflow even for a finite
        // logit, and an out-of-range float-to-int conversion is undefined.
        value = std::min(value * scale,
                         static_cast<float>(config_.duration_max_duration));
        value = static_cast<float>(sano_round_half_even(value));
        value = std::max(1.0f, std::min(
            value, static_cast<float>(config_.duration_max_duration)));
        durations[index] = static_cast<int32_t>(value);
        total_frames += durations[index];
        if (total_frames > INT32_MAX) return {};
    }
    return durations;
}

SanoPiperFrontOutput SanoPiperModel::predict_latent(
    ModelLoader& loader,
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& durations,
    ggml_backend_t backend
) const {
    SanoPiperFrontOutput output;
    if (token_ids.empty() || durations.size() != token_ids.size()) return output;
    int64_t frames64 = 0;
    for (int32_t duration : durations) {
        if (duration < 1) return output;
        frames64 += duration;
        if (frames64 > INT32_MAX) return output;
    }
    const int frames = static_cast<int>(frames64);
#if INFLECT_SANO_MAX_FRAMES > 0
    if (frames > INFLECT_SANO_MAX_FRAMES) {
        std::fprintf(
            stderr,
            "[SanoPiper] acoustic frame cap exceeded: %d > %d; "
            "split text into shorter phrases\n",
            frames, INFLECT_SANO_MAX_FRAMES);
        return output;
    }
#endif

    std::vector<float> context =
        run_token_context(loader, config_, token_ids, durations, backend);
    if (context.empty()) return output;
    std::vector<float> latent = run_frame_stage(
        loader, config_, context, durations, frames, backend);
    if (latent.size() != static_cast<size_t>(frames) *
                             config_.acoustic_out_channels) {
        return output;
    }
    output.durations = durations;
    output.latent = std::move(latent);
    output.frames = frames;
    output.channels = config_.acoustic_out_channels;
    return output;
}

std::vector<float> SanoPiperModel::synthesize(
    ModelLoader& loader,
    const std::vector<int32_t>& token_ids,
    float speaking_rate,
    ggml_backend_t backend
) {
    packed_conv::profile_reset();
    const uint32_t started = runtime_now_ms();
    // Amy's ~880 KB GGUF fits alongside bounded decoder graphs. One sequential
    // read also makes later utterances independent of SD-card latency.
    loader.cache_weights(1024 * 1024);
    if (!loader.select({"duration."})) return {};
    std::vector<int32_t> durations =
        predict_durations(loader, token_ids, speaking_rate, backend);
    loader.release_selected();
    if (durations.empty()) return {};

    if (!loader.select({"acoustic."})) return {};
    SanoPiperFrontOutput front =
        predict_latent(loader, token_ids, durations, backend);
    loader.release_selected();
    if (front.latent.empty() || front.frames <= 0) return {};

    std::vector<float> audio = decoder_.decode(
        loader, std::move(front.latent), front.frames, backend);
    packed_conv::profile_log(runtime_now_ms() - started);
    return audio;
}

} // namespace inflect
