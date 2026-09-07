#include "sano_piper_decoder.h"

#include "memory_trace.h"

#include "../ggml/include/ggml-alloc.h"

#include <algorithm>
#include <cstdio>
#include <new>
#include <string>
#include <utility>

namespace inflect {
namespace {

constexpr int kBankKernels[3] = {3, 5, 7};
constexpr int kBankDilations1[3] = {1, 2, 3};
constexpr int kBankDilations2[3] = {2, 6, 12};

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

bool valid_conv_transpose(
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
    // PyTorch ConvTranspose1d [in, out, K] is represented by GGML as
    // ne=[K,out,in].
    return weight->ne[0] == kernel && weight->ne[1] == out_channels &&
           weight->ne[2] == in_channels;
}

bool valid_scalar(const ggml_tensor* value) {
    return value != nullptr && ggml_nelements(value) == 1;
}

ggml_tensor* scalar_f32(ggml_context* ctx, ggml_tensor* value) {
    if (!value) return nullptr;
    return value->type == GGML_TYPE_F32
               ? value
               : ggml_cast(ctx, value, GGML_TYPE_F32);
}

ggml_context* new_stage_context(size_t bytes = 192 * 1024) {
    struct ggml_init_params params = {
        .mem_size = bytes,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    return ggml_init(params);
}

SanoBuffer execute_stage(
    ggml_gallocr_t& allocator,
    ggml_context* ctx,
    ggml_cgraph* graph,
    ggml_tensor* input,
    const float* input_data,
    int full_frames,
    int offset,
    ggml_tensor* output,
    ggml_backend_t backend,
    const char* label
) {
    SanoBuffer result;
    if (!ctx || !graph || !input || !output || !backend ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !input_data || offset < 0 || offset + input->ne[0] > full_frames) {
        return result;
    }

    if (!allocator) {
        allocator = ggml_gallocr_new(runtime_weight_buffer_type());
        if (!allocator || !ggml_gallocr_reserve(allocator, graph) ||
            !ggml_gallocr_alloc_graph(allocator, graph)) {
            std::fprintf(stderr, "[SanoDecoder] failed to allocate %s graph\n", label);
            if (allocator) ggml_gallocr_free(allocator);
            allocator = nullptr;
            return result;
        }
        mem_trace_graph(label, ctx, allocator);
        std::fprintf(stderr, "[SanoGraph] %s frames=%lld graph_bytes=%zu\n",
                     label, static_cast<long long>(input->ne[0]),
                     ggml_gallocr_get_buffer_size(allocator, 0));
    }
    for (int channel = 0; channel < input->ne[1]; ++channel) {
        ggml_backend_tensor_set(input,
            input_data + static_cast<size_t>(channel) * full_frames + offset,
            channel * input->nb[1], input->ne[0] * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "[SanoDecoder] %s graph failed: %s\n",
                     label, ggml_status_to_string(status));
        return result;
    }

    if (!result.allocate(static_cast<size_t>(ggml_nelements(output)))) {
        return {};
    }
    ggml_backend_tensor_get(
        output, result.data(), 0, result.size() * sizeof(float));
#if defined(INFLECT_LOW_MEMORY)
    mem_release_to_os();
#endif
    return result;
}

ggml_tensor* build_residual_bank(
    ggml_context* ctx,
    ModelLoader& loader,
    const SanoPiperConfig& config,
    int stage,
    ggml_tensor* input,
    std::vector<VocoderQuantConv1dOpData>& conv_ops
) {
    const int channels = config.decoder_channels[stage + 1];
    ggml_tensor* sum = nullptr;
    int active = 0;
    ggml_tensor* activated = ggml_leaky_relu(ctx, input, 0.1f, false);
    for (int branch = 0; branch < 3; ++branch) {
        if (!config.branch_enabled(stage, branch)) continue;
        const std::string root =
            "decoder.res" + std::to_string(stage) + ".0.blocks." +
            std::to_string(branch) + ".";
        ggml_tensor* conv1_w = loader.get_tensor(root + "conv1.weight");
        ggml_tensor* conv1_b = loader.get_tensor(root + "conv1.bias");
        ggml_tensor* conv2_w = loader.get_tensor(root + "conv2.weight");
        ggml_tensor* conv2_b = loader.get_tensor(root + "conv2.bias");
        const int kernel = kBankKernels[branch];
        if (!valid_conv(conv1_w, conv1_b, channels, channels, kernel) ||
            !valid_conv(conv2_w, conv2_b, channels, channels, kernel)) {
            std::fprintf(stderr,
                         "[SanoDecoder] invalid residual bank tensor %s\n",
                         root.c_str());
            return nullptr;
        }

        ggml_tensor* value = activated;
        value = packed_conv::conv1d(
            ctx, conv1_w, conv1_b, value, kernel, 1,
            kBankDilations1[branch] * (kernel / 2),
            kBankDilations1[branch], conv_ops, "sano.res.conv1");
        ggml_tensor* y1 = ggml_add(ctx, value, input);
        value = ggml_leaky_relu(ctx, y1, 0.1f, false);
        value = packed_conv::conv1d(
            ctx, conv2_w, conv2_b, value, kernel, 1,
            kBankDilations2[branch] * (kernel / 2),
            kBankDilations2[branch], conv_ops, "sano.res.conv2");
        ggml_tensor* y2 = ggml_add(ctx, value, y1);
        sum = sum == nullptr ? y2 : ggml_add_inplace(ctx, sum, y2);
        ++active;
    }
    if (!sum || active == 0) return nullptr;
    return active == 1 ? sum : ggml_scale_inplace(ctx, sum, 1.0f / active);
}

ggml_tensor* build_post_filter(
    ggml_context* ctx,
    ModelLoader& loader,
    const SanoPiperConfig& config,
    ggml_tensor* audio,
    std::vector<VocoderQuantConv1dOpData>& conv_ops
) {
    if (config.post_filter_channels <= 0) return audio;
    const int channels = config.post_filter_channels;
    ggml_tensor* in_w = loader.get_tensor("decoder.post_filter.in_conv.weight");
    ggml_tensor* in_b = loader.get_tensor("decoder.post_filter.in_conv.bias");
    if (!valid_conv(in_w, in_b, 1, channels, config.post_filter_kernel)) {
        return nullptr;
    }
    ggml_tensor* value = packed_conv::conv1d(
        ctx, in_w, in_b, audio, config.post_filter_kernel, 1,
        config.post_filter_kernel / 2, 1, conv_ops, "sano.post_filter.in");

    for (int layer = 0; layer < config.post_filter_layers; ++layer) {
        const std::string root =
            "decoder.post_filter.units." + std::to_string(layer) + ".";
        ggml_tensor* scale = loader.get_tensor(root + "scale");
        ggml_tensor* conv1_w = loader.get_tensor(root + "conv1.weight");
        ggml_tensor* conv1_b = loader.get_tensor(root + "conv1.bias");
        ggml_tensor* conv2_w = loader.get_tensor(root + "conv2.weight");
        ggml_tensor* conv2_b = loader.get_tensor(root + "conv2.bias");
        if (!valid_scalar(scale) ||
            !valid_conv(conv1_w, conv1_b, channels, channels, 3) ||
            !valid_conv(conv2_w, conv2_b, channels, channels, 3)) {
            return nullptr;
        }
        ggml_tensor* branch = ggml_leaky_relu(ctx, value, 0.1f, false);
        const int dilation = 1 + layer;
        branch = packed_conv::conv1d(
            ctx, conv1_w, conv1_b, branch, 3, 1, dilation, dilation,
            conv_ops, "sano.post_filter.conv1");
        branch = ggml_leaky_relu(ctx, branch, 0.1f, false);
        branch = packed_conv::conv1d(
            ctx, conv2_w, conv2_b, branch, 3, 1, 1, 1,
            conv_ops, "sano.post_filter.conv2");
        branch = ggml_mul(ctx, branch, scalar_f32(ctx, scale));
        value = ggml_add(ctx, value, branch);
    }

    ggml_tensor* out_w = loader.get_tensor("decoder.post_filter.out_conv.weight");
    ggml_tensor* out_b = loader.get_tensor("decoder.post_filter.out_conv.bias");
    if (!valid_conv(out_w, out_b, channels, 1, config.post_filter_kernel)) {
        return nullptr;
    }
    ggml_tensor* correction = packed_conv::conv1d(
        ctx, out_w, out_b, value, config.post_filter_kernel, 1,
        config.post_filter_kernel / 2, 1, conv_ops, "sano.post_filter.out");
    correction = ggml_scale(ctx, correction, config.post_filter_scale);
    return ggml_tanh(ctx, ggml_add(ctx, audio, correction));
}

} // namespace

SanoBuffer SanoPiperDecoder::run_pre(
    GraphState& state,
    ModelLoader& loader,
    const float* input,
    int full_frames,
    int offset,
    int frames,
    ggml_backend_t backend
) {
    SanoBuffer empty;
    const int in_channels = config_.acoustic_out_channels;
    const int out_channels = config_.decoder_channels[0];
    if (frames <= 0 || !input)
        return empty;
    if (state.ctx && state.input->ne[0] == frames) {
        return execute_stage(state.allocator, state.ctx, state.graph, state.input,
            input, full_frames, offset, state.output, backend, "sano.pre");
    }
    state.clear();

    ggml_tensor* weight = loader.get_tensor("decoder.pre.weight");
    ggml_tensor* bias = loader.get_tensor("decoder.pre.bias");
    if (!valid_conv(weight, bias, in_channels, out_channels,
                    config_.decoder_pre_kernel)) {
        std::fprintf(stderr, "[SanoDecoder] invalid decoder.pre tensors\n");
        return empty;
    }

    ggml_context* ctx = new_stage_context();
    if (!ctx) return empty;
    std::vector<VocoderQuantConv1dOpData> conv_ops;
    conv_ops.reserve(1);
    ggml_tensor* source =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, in_channels, 1);
    ggml_set_input(source);
    ggml_tensor* value = packed_conv::conv1d(
        ctx, weight, bias, source, config_.decoder_pre_kernel, 1,
        config_.decoder_pre_kernel / 2, 1, conv_ops, "sano.decoder.pre");
    ggml_set_output(value);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, value);
    state.ctx = ctx;
    state.graph = graph;
    state.input = source;
    state.output = value;
    state.conv_ops = std::move(conv_ops);
    return execute_stage(state.allocator, ctx, graph, source, input,
                         full_frames, offset, value, backend, "sano.pre");
}

SanoBuffer SanoPiperDecoder::run_stage(
    GraphState& state,
    ModelLoader& loader,
    const float* input,
    int full_frames,
    int offset,
    int input_frames,
    int stage,
    ggml_backend_t backend
) {
    SanoBuffer empty;
    if (stage < 0 || stage >= 3 || input_frames <= 0) return empty;
    const int in_channels = config_.decoder_channels[stage];
    const int out_channels = config_.decoder_channels[stage + 1];
    if (!input)
        return empty;
    const std::string label = "sano.stage" + std::to_string(stage);
    if (state.ctx && state.input->ne[0] == input_frames) {
        return execute_stage(state.allocator, state.ctx, state.graph, state.input,
            input, full_frames, offset, state.output, backend, label.c_str());
    }
    state.clear();

    const std::string up = "decoder.up" + std::to_string(stage) + ".";
    ggml_tensor* up_w = loader.get_tensor(up + "weight");
    ggml_tensor* up_b = loader.get_tensor(up + "bias");
    const int kernel = config_.decoder_up_kernels[stage];
    const int stride = config_.decoder_up_strides[stage];
    const int padding = config_.decoder_up_paddings[stage];
    if (!valid_conv_transpose(up_w, up_b, in_channels, out_channels, kernel)) {
        std::fprintf(stderr, "[SanoDecoder] invalid %s tensors\n", up.c_str());
        return empty;
    }

    ggml_context* ctx = new_stage_context(256 * 1024);
    if (!ctx) return empty;
    std::vector<VocoderQuantConv1dOpData> conv_ops;
    conv_ops.reserve(static_cast<size_t>(config_.branch_count(stage)) * 2);
    std::vector<QuantConvTranspose1dOpData> transpose_ops;
    transpose_ops.reserve(1);

    ggml_tensor* source = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, input_frames, in_channels, 1);
    ggml_set_input(source);
    ggml_tensor* value = ggml_leaky_relu(ctx, source, 0.1f, false);
    value = packed_conv::conv_transpose1d(
        ctx, up_w, value, kernel, stride, padding,
        transpose_ops, "sano.decoder.upsample");
    value = packed_conv::add_channel_bias(ctx, value, up_b);
    value = build_residual_bank(ctx, loader, config_, stage, value, conv_ops);
    if (!value) {
        ggml_free(ctx);
        return empty;
    }
    ggml_set_output(value);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, value);
    state.ctx = ctx;
    state.graph = graph;
    state.input = source;
    state.output = value;
    state.conv_ops = std::move(conv_ops);
    state.transpose_ops = std::move(transpose_ops);
    return execute_stage(state.allocator, ctx, graph, source, input,
                         full_frames, offset, value, backend, label.c_str());
}

SanoBuffer SanoPiperDecoder::run_post(
    GraphState& state,
    ModelLoader& loader,
    const float* input,
    int full_frames,
    int offset,
    int frames,
    ggml_backend_t backend
) {
    SanoBuffer empty;
    const int channels = config_.decoder_channels[3];
    if (frames <= 0 || !input)
        return empty;
    if (state.ctx && state.input->ne[0] == frames) {
        return execute_stage(state.allocator, state.ctx, state.graph, state.input,
            input, full_frames, offset, state.output, backend, "sano.post");
    }
    state.clear();

    ggml_tensor* post_w = loader.get_tensor("decoder.post.weight");
    ggml_tensor* post_b = loader.get_tensor("decoder.post.bias");
    if (!valid_conv(post_w, post_b, channels, 1, config_.decoder_post_kernel)) {
        std::fprintf(stderr, "[SanoDecoder] invalid decoder.post tensors\n");
        return empty;
    }

    ggml_context* ctx = new_stage_context(256 * 1024);
    if (!ctx) return empty;
    std::vector<VocoderQuantConv1dOpData> conv_ops;
    conv_ops.reserve(1 + static_cast<size_t>(config_.post_filter_layers) * 2 + 2);
    ggml_tensor* source =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, channels, 1);
    ggml_set_input(source);
    ggml_tensor* value = ggml_leaky_relu(ctx, source, 0.01f, false);
    value = packed_conv::conv1d(
        ctx, post_w, post_b, value, config_.decoder_post_kernel, 1,
        config_.decoder_post_kernel / 2, 1, conv_ops, "sano.decoder.post");
    value = ggml_tanh(ctx, value);
    value = build_post_filter(ctx, loader, config_, value, conv_ops);
    if (!value) {
        ggml_free(ctx);
        return empty;
    }
    ggml_set_output(value);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, value);
    state.ctx = ctx;
    state.graph = graph;
    state.input = source;
    state.output = value;
    state.conv_ops = std::move(conv_ops);
    return execute_stage(state.allocator, ctx, graph, source, input,
                         full_frames, offset, value, backend, "sano.post");
}

std::vector<float> SanoPiperDecoder::decode(
    ModelLoader& loader,
    std::vector<float> latent,
    int frames,
    ggml_backend_t backend,
    bool tiled
) {
    if (frames <= 0 ||
        latent.size() != static_cast<size_t>(frames) *
                             config_.acoustic_out_channels) {
        return {};
    }

    // The converter writes tensors in this exact runtime order so each select
    // becomes one bulk file range on host/ESP backends when alignment permits.
    packed_conv::Q4WeightCache weight_cache;
    GraphState graph_state;
    if (!loader.select({"decoder.pre."})) return {};
    SanoBuffer value = sano_run_tiled(
        latent.data(), frames, config_.acoustic_out_channels,
        config_.decoder_channels[0], 1, tiled ? 64 : frames, config_.decoder_pre_kernel / 2,
        "decoder.pre", [&](const float* data, int full, int offset, int count) {
            return run_pre(graph_state, loader, data, full, offset, count, backend);
        });
    graph_state.clear();
    std::vector<float>().swap(latent);
    loader.release_selected();
    if (value.empty()) return {};

    int current_frames = frames;
    for (int stage = 0; stage < 3; ++stage) {
        weight_cache.clear();
        if (!loader.select({
                "decoder.up" + std::to_string(stage) + ".",
                "decoder.res" + std::to_string(stage) + "."})) {
            return {};
        }
        int radius = 0;
        for (int branch = 0; branch < 3; ++branch) {
            if (config_.branch_enabled(stage, branch)) {
                radius = std::max(radius, (kBankKernels[branch] / 2) *
                    (kBankDilations1[branch] + kBankDilations2[branch]));
            }
        }
        const int rate = config_.decoder_up_strides[stage];
        const int halo = (radius + config_.decoder_up_kernels[stage] + rate - 1) / rate;
        // The last stage works at audio rate. Keep its graph in a small PSRAM
        // working set instead of retaining many utterance-sized residuals.
        // Spend ~1 MiB of graph memory on Amy's narrow final stage to reduce
        // halo work. Scale the window down for models with wider channels.
        const int cores[] = {32, 64,
            std::max(32, std::min<int>(384, 9216 / config_.decoder_channels[3]))};
        value = sano_run_tiled(
            value.data(), current_frames, config_.decoder_channels[stage],
            config_.decoder_channels[stage + 1], rate, tiled ? cores[stage] : current_frames, halo,
            ("decoder.stage" + std::to_string(stage)).c_str(),
            [&](const float* data, int full, int offset, int count) {
                return run_stage(graph_state, loader, data, full, offset, count, stage, backend);
            });
        graph_state.clear();
        loader.release_selected();
        if (value.empty()) return {};
        current_frames *= config_.decoder_up_strides[stage];
    }

    std::vector<std::string> post_prefixes{"decoder.post."};
    weight_cache.clear();
    if (config_.post_filter_channels > 0) {
        post_prefixes.push_back("decoder.post_filter.");
    }
    if (!loader.select(post_prefixes)) return {};
    int post_halo = config_.decoder_post_kernel / 2;
    if (config_.post_filter_channels > 0) {
        post_halo += config_.post_filter_kernel - 1;
        for (int layer = 0; layer < config_.post_filter_layers; ++layer) {
            post_halo += layer + 2;
        }
    }
    value = sano_run_tiled(
        value.data(), current_frames, config_.decoder_channels[3], 1, 1,
        tiled ? (config_.post_filter_channels > 0 ? 512 :
            std::max(64, std::min<int>(4096, 98304 / config_.decoder_channels[3]))) : current_frames,
        post_halo, "decoder.post",
        [&](const float* data, int full, int offset, int count) {
            return run_post(graph_state, loader, data, full, offset, count, backend);
        });
    graph_state.clear();
    loader.release_selected();
    if (value.size() != static_cast<size_t>(frames) *
                            SanoPiperConfig::kHopLength) {
        std::fprintf(stderr,
                     "[SanoDecoder] output length mismatch: got=%zu expected=%zu\n",
                     value.size(),
                     static_cast<size_t>(frames) * SanoPiperConfig::kHopLength);
        return {};
    }
    // The large activation graphs and stage inputs have been released before
    // constructing the public audio vector.
#if defined(__cpp_exceptions)
    try {
#endif
        return std::vector<float>(value.data(), value.data() + value.size());
#if defined(__cpp_exceptions)
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "[SanoMemory] public audio allocation failed samples=%zu\n", value.size());
        runtime_trace_heap("sano audio allocation failed");
        return {};
    }
#endif
}

} // namespace inflect
