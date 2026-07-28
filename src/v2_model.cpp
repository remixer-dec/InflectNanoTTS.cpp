#include "v2_model.h"

#include "utils.h"
#include "v2_symbols.h"

#include "../ggml/include/ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

namespace inflect {
namespace {

std::string debug_directory() {
    const char* value = std::getenv("INFLECT_DUMP_DIR");
    return value && value[0] ? value : "";
}

std::string flat_debug_name(std::string name) {
    for (char& c : name) {
        if (c == '/') c = '_';
    }
    return name;
}

void debug_record(const std::string& name, const char* dtype,
                  const std::string& shape, size_t count,
                  const void* data, size_t bytes) {
    const std::string directory = debug_directory();
    if (directory.empty()) return;
    std::filesystem::create_directories(directory);
    const std::string file = flat_debug_name(name) + "." + dtype;
    std::ofstream payload(directory + "/" + file, std::ios::binary);
    payload.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    std::ofstream manifest(directory + "/v2_manifest.tsv", std::ios::app);
    manifest << name << "\t" << dtype << "\t" << shape << "\t"
             << count << "\t" << file << "\n";
}

void debug_time_major(const std::string& name, const std::vector<float>& value,
                      int time, int channels) {
    if (debug_directory().empty()) return;
    std::vector<float> channel_major(value.size());
    for (int c = 0; c < channels; ++c) {
        for (int t = 0; t < time; ++t) {
            channel_major[static_cast<size_t>(c) * time + t] =
                value[static_cast<size_t>(t) * channels + c];
        }
    }
    debug_record(name, "f32",
                 "[1," + std::to_string(channels) + "," +
                     std::to_string(time) + "]",
                 channel_major.size(), channel_major.data(),
                 channel_major.size() * sizeof(float));
}

void debug_vector(const std::string& name, const std::vector<float>& value,
                  const std::string& shape) {
    if (debug_directory().empty()) return;
    debug_record(name, "f32", shape, value.size(), value.data(),
                 value.size() * sizeof(float));
}

struct TensorRow {
    std::vector<float> values;

    const float* read(const ggml_tensor* tensor, int64_t row, int64_t count) {
        if (!tensor || count <= 0) return nullptr;
        const int dims = ggml_n_dims(tensor);
        const int64_t rows = ggml_nelements(tensor) / tensor->ne[0];
        // For normal Conv1d tensors a logical output row spans ne[0]*ne[1]
        // and is selected in dimension 2.
        int64_t row_width = tensor->ne[0];
        size_t byte_offset = static_cast<size_t>(row) * tensor->nb[1];
        const bool singleton_output_conv =
            dims == 2 && tensor->ne[2] == 1 && count > tensor->ne[0] &&
            count <= tensor->ne[0] * tensor->ne[1];
        if (dims == 3 || singleton_output_conv) {
            row_width *= tensor->ne[1];
            byte_offset = static_cast<size_t>(row) * tensor->nb[2];
        }
        if (count > row_width || row < 0 ||
            (dims == 2 && !singleton_output_conv && row >= rows) ||
            (dims == 3 && row >= tensor->ne[2])) {
            return nullptr;
        }
        const char* source = static_cast<const char*>(tensor->data) + byte_offset;
        if (tensor->type == GGML_TYPE_F32) {
            return reinterpret_cast<const float*>(source);
        }
        values.resize(static_cast<size_t>(row_width));
        if (tensor->type == GGML_TYPE_F16) {
            const auto* half = reinterpret_cast<const ggml_fp16_t*>(source);
            for (int64_t i = 0; i < row_width; ++i) {
                values[static_cast<size_t>(i)] = ggml_fp16_to_fp32(half[i]);
            }
            return values.data();
        }
        const auto* traits = ggml_get_type_traits(tensor->type);
        if (!traits || !traits->to_float || dims > 2) return nullptr;
        traits->to_float(source, values.data(), row_width);
        return values.data();
    }
};

float scalar_at(const ggml_tensor* tensor, int64_t index) {
    if (!tensor || index < 0 || index >= ggml_nelements(tensor)) return 0.0f;
    if (tensor->type == GGML_TYPE_F32) {
        return static_cast<const float*>(tensor->data)[index];
    }
    if (tensor->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(
            static_cast<const ggml_fp16_t*>(tensor->data)[index]);
    }
    return 0.0f;
}

bool conv1d(
    const ggml_tensor* weight,
    const ggml_tensor* bias,
    const std::vector<float>& input,
    int time,
    int in_channels,
    int out_channels,
    int kernel,
    int dilation,
    std::vector<float>& output
) {
    if (!weight || static_cast<int64_t>(input.size()) !=
                       static_cast<int64_t>(time) * in_channels) {
        return false;
    }
    const bool native_conv_shape =
        weight->ne[0] == kernel && weight->ne[1] == in_channels &&
        weight->ne[2] == out_channels;
    const bool flattened = !native_conv_shape;
    if ((!flattened &&
         (weight->ne[0] != kernel || weight->ne[1] != in_channels ||
          weight->ne[2] != out_channels)) ||
        (flattened &&
         (weight->ne[0] < kernel * in_channels ||
          weight->ne[1] != out_channels))) {
        std::fprintf(stderr,
                     "[V2Model] incompatible convolution %s shape=[%lld,%lld,%lld]\n",
                     weight->name,
                     static_cast<long long>(weight->ne[0]),
                     static_cast<long long>(weight->ne[1]),
                     static_cast<long long>(weight->ne[2]));
        return false;
    }
    output.assign(static_cast<size_t>(time) * out_channels, 0.0f);
    const int padding = (kernel * dilation - dilation) / 2;
    TensorRow row;
    for (int out = 0; out < out_channels; ++out) {
        const float* w = row.read(weight, out, kernel * in_channels);
        if (!w) return false;
        const float b = bias ? scalar_at(bias, out) : 0.0f;
        for (int t = 0; t < time; ++t) {
            float sum = b;
            for (int k = 0; k < kernel; ++k) {
                const int source_t = t + k * dilation - padding;
                if (source_t < 0 || source_t >= time) continue;
                for (int in = 0; in < in_channels; ++in) {
                    const int weight_index = k + kernel * in;
                    sum += input[static_cast<size_t>(source_t) * in_channels + in] *
                           w[weight_index];
                }
            }
            output[static_cast<size_t>(t) * out_channels + out] = sum;
        }
    }
    return true;
}

bool layer_norm(
    const ggml_tensor* gamma,
    const ggml_tensor* beta,
    std::vector<float>& value,
    int time,
    int channels
) {
    if (!gamma || !beta ||
        static_cast<int64_t>(value.size()) != static_cast<int64_t>(time) * channels) {
        return false;
    }
    for (int t = 0; t < time; ++t) {
        float mean = 0.0f;
        for (int c = 0; c < channels; ++c) {
            mean += value[static_cast<size_t>(t) * channels + c];
        }
        mean /= channels;
        float variance = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const float d = value[static_cast<size_t>(t) * channels + c] - mean;
            variance += d * d;
        }
        const float inv = 1.0f / std::sqrt(variance / channels + 1e-5f);
        for (int c = 0; c < channels; ++c) {
            float& x = value[static_cast<size_t>(t) * channels + c];
            x = (x - mean) * inv * scalar_at(gamma, c) + scalar_at(beta, c);
        }
    }
    return true;
}

void add_in_place(std::vector<float>& dst, const std::vector<float>& src) {
    for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
}

bool attention(
    ModelLoader& loader,
    int layer,
    const std::vector<float>& input,
    int time,
    int hidden_channels,
    int heads,
    std::vector<float>& output
) {
    if (hidden_channels <= 0 || heads <= 0 ||
        hidden_channels % heads != 0) {
        return false;
    }
    const int head_channels = hidden_channels / heads;
    const std::string root =
        "enc_p.encoder.attn_layers." + std::to_string(layer) + ".";
    std::vector<float> q, k, v;
    if (!conv1d(loader.get_tensor(root + "conv_q.weight"),
                loader.get_tensor(root + "conv_q.bias"),
                input, time, hidden_channels, hidden_channels, 1, 1, q) ||
        !conv1d(loader.get_tensor(root + "conv_k.weight"),
                loader.get_tensor(root + "conv_k.bias"),
                input, time, hidden_channels, hidden_channels, 1, 1, k) ||
        !conv1d(loader.get_tensor(root + "conv_v.weight"),
                loader.get_tensor(root + "conv_v.bias"),
                input, time, hidden_channels, hidden_channels, 1, 1, v)) {
        return false;
    }
    const ggml_tensor* rel_k = loader.get_tensor(root + "emb_rel_k");
    const ggml_tensor* rel_v = loader.get_tensor(root + "emb_rel_v");
    if (!rel_k || !rel_v ||
        rel_k->ne[0] != head_channels || rel_k->ne[1] != 9) {
        return false;
    }
    std::vector<float> attended(
        static_cast<size_t>(time) * hidden_channels, 0.0f);
    std::vector<float> scores(time);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_channels));
    TensorRow rel_k_row;
    TensorRow rel_v_row;
    for (int head = 0; head < heads; ++head) {
        const int channel_offset = head * head_channels;
        for (int query_t = 0; query_t < time; ++query_t) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (int key_t = 0; key_t < time; ++key_t) {
                float score = 0.0f;
                for (int c = 0; c < head_channels; ++c) {
                    score += q[static_cast<size_t>(query_t) * hidden_channels +
                               channel_offset + c] *
                             k[static_cast<size_t>(key_t) * hidden_channels +
                               channel_offset + c];
                }
                score *= scale;
                const int relative = key_t - query_t;
                if (relative >= -4 && relative <= 4) {
                    const float* embedding =
                        rel_k_row.read(rel_k, relative + 4, head_channels);
                    if (!embedding) return false;
                    float relative_score = 0.0f;
                    for (int c = 0; c < head_channels; ++c) {
                        relative_score +=
                            q[static_cast<size_t>(query_t) * hidden_channels +
                              channel_offset + c] * embedding[c];
                    }
                    score += relative_score * scale;
                }
                scores[key_t] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int key_t = 0; key_t < time; ++key_t) {
                const float probability = scores[key_t] / denominator;
                const int relative = key_t - query_t;
                const float* relative_value = nullptr;
                if (relative >= -4 && relative <= 4) {
                    relative_value =
                        rel_v_row.read(rel_v, relative + 4, head_channels);
                    if (!relative_value) return false;
                }
                for (int c = 0; c < head_channels; ++c) {
                    float value =
                        v[static_cast<size_t>(key_t) * hidden_channels +
                          channel_offset + c];
                    if (relative_value) value += relative_value[c];
                    attended[static_cast<size_t>(query_t) * hidden_channels +
                             channel_offset + c] += probability * value;
                }
            }
        }
    }
    return conv1d(loader.get_tensor(root + "conv_o.weight"),
                  loader.get_tensor(root + "conv_o.bias"),
                  attended, time, hidden_channels, hidden_channels, 1, 1, output);
}

bool reverse_coupling(
    ModelLoader& loader,
    int flow_index,
    std::vector<float>& latent,
    int time,
    int latent_channels,
    int hidden_channels
) {
    if (latent_channels <= 0 || latent_channels % 2 != 0 ||
        hidden_channels <= 0) {
        return false;
    }
    const int half_channels = latent_channels / 2;
    const std::string root = "flow.flows." + std::to_string(flow_index) + ".";
    std::vector<float> first(static_cast<size_t>(time) * half_channels);
    for (int t = 0; t < time; ++t) {
        std::copy_n(latent.data() + static_cast<size_t>(t) * latent_channels,
                    half_channels,
                    first.data() + static_cast<size_t>(t) * half_channels);
    }
    std::vector<float> hidden;
    if (!conv1d(loader.get_tensor(root + "pre.weight"),
                loader.get_tensor(root + "pre.bias"),
                first, time, half_channels, hidden_channels, 1, 1, hidden)) {
        return false;
    }
    std::vector<float> skip(
        static_cast<size_t>(time) * hidden_channels, 0.0f);
    for (int layer = 0; layer < 4; ++layer) {
        const std::string enc =
            root + "enc.in_layers." + std::to_string(layer) + ".";
        std::vector<float> gates;
        if (!conv1d(loader.get_tensor(enc + "weight"),
                    loader.get_tensor(enc + "bias"),
                    hidden, time, hidden_channels, hidden_channels * 2,
                    5, 1, gates)) {
            return false;
        }
        std::vector<float> acts(
            static_cast<size_t>(time) * hidden_channels);
        for (int t = 0; t < time; ++t) {
            for (int c = 0; c < hidden_channels; ++c) {
                const float a =
                    gates[static_cast<size_t>(t) * hidden_channels * 2 + c];
                const float b =
                    gates[static_cast<size_t>(t) * hidden_channels * 2 +
                          hidden_channels + c];
                acts[static_cast<size_t>(t) * hidden_channels + c] =
                    std::tanh(a) / (1.0f + std::exp(-b));
            }
        }
        const int output_channels =
            layer < 3 ? hidden_channels * 2 : hidden_channels;
        const std::string res =
            root + "enc.res_skip_layers." + std::to_string(layer) + ".";
        std::vector<float> projected;
        if (!conv1d(loader.get_tensor(res + "weight"),
                    loader.get_tensor(res + "bias"),
                    acts, time, hidden_channels, output_channels,
                    1, 1, projected)) {
            return false;
        }
        if (layer < 3) {
            for (int t = 0; t < time; ++t) {
                for (int c = 0; c < hidden_channels; ++c) {
                    hidden[static_cast<size_t>(t) * hidden_channels + c] +=
                        projected[static_cast<size_t>(t) * output_channels + c];
                    skip[static_cast<size_t>(t) * hidden_channels + c] +=
                        projected[static_cast<size_t>(t) * output_channels +
                                  hidden_channels + c];
                }
            }
        } else {
            add_in_place(skip, projected);
        }
    }
    std::vector<float> mean;
    if (!conv1d(loader.get_tensor(root + "post.weight"),
                loader.get_tensor(root + "post.bias"),
                skip, time, hidden_channels, half_channels, 1, 1, mean)) {
        return false;
    }
    for (int t = 0; t < time; ++t) {
        for (int c = 0; c < half_channels; ++c) {
            latent[static_cast<size_t>(t) * latent_channels +
                   half_channels + c] -=
                mean[static_cast<size_t>(t) * half_channels + c];
        }
    }
    return true;
}

void flip_channels(std::vector<float>& latent, int time, int channels) {
    for (int t = 0; t < time; ++t) {
        for (int c = 0; c < channels / 2; ++c) {
            std::swap(latent[static_cast<size_t>(t) * channels + c],
                      latent[static_cast<size_t>(t) * channels +
                             (channels - 1 - c)]);
        }
    }
}

} // namespace

V2Model::V2Model() = default;

ggml_tensor* V2Model::tensor(const std::string& name) const {
    return loader_ ? loader_->get_tensor(name) : nullptr;
}

bool V2Model::validate_tensor(const std::string& name) const {
    if (!loader_ || !loader_->get_tensor(name)) {
        std::fprintf(stderr, "[V2Model] Missing required tensor: %s\n", name.c_str());
        return false;
    }
    return true;
}

bool V2Model::configure_architecture(ModelLoader& loader) {
    static constexpr const char* required_metadata[] = {
        "general.architecture", "inflect.v2.symbol_hash",
        "inflect.v2.sample_rate", "inflect.v2.inter_channels",
        "inflect.v2.hidden_channels", "inflect.v2.filter_channels",
        "inflect.v2.n_heads", "inflect.v2.n_layers",
        "inflect.v2.upsample_initial_channel",
    };
    for (const char* key : required_metadata) {
        if (!loader.has_key(key)) {
            std::fprintf(stderr, "[V2Model] Missing required GGUF metadata: %s\n", key);
            return false;
        }
    }
    if (loader.get_string("general.architecture") != "inflect-v2" ||
        loader.get_string("inflect.v2.symbol_hash") != v2::kSymbolHashHex) {
        std::fprintf(stderr, "[V2Model] Incompatible v2 architecture metadata\n");
        return false;
    }
    const int sample_rate = loader.get_i32("inflect.v2.sample_rate");
    const int latent = loader.get_i32("inflect.v2.inter_channels");
    const int hidden = loader.get_i32("inflect.v2.hidden_channels");
    const int filter = loader.get_i32("inflect.v2.filter_channels");
    const int heads = loader.get_i32("inflect.v2.n_heads");
    const int layers = loader.get_i32("inflect.v2.n_layers");
    const int decoder_width =
        loader.get_i32("inflect.v2.upsample_initial_channel");
    if (sample_rate != 24000 || latent <= 0 || latent > 512 ||
        latent % 2 != 0 || hidden <= 0 || hidden > 512 ||
        heads <= 0 || hidden % heads != 0 ||
        filter <= 0 || filter > 4096 ||
        layers <= 0 || layers > 16 ||
        decoder_width <= 0 || decoder_width > 2048 ||
        decoder_width % 16 != 0) {
        std::fprintf(
            stderr,
            "[V2Model] Invalid v2 dimensions latent=%d hidden=%d filter=%d "
            "heads=%d layers=%d decoder=%d sample_rate=%d\n",
            latent, hidden, filter, heads, layers, decoder_width, sample_rate);
        return false;
    }
    sample_rate_ = sample_rate;
    latent_channels_ = latent;
    hidden_channels_ = hidden;
    filter_channels_ = filter;
    attention_heads_ = heads;
    encoder_layers_ = layers;
    upsample_initial_channels_ = decoder_width;
    return true;
}

bool V2Model::load_duration(ModelLoader& loader) {
    if (!configure_architecture(loader)) return false;
    loader_ = &loader;
    static constexpr const char* required[] = {
        "enc_p.emb.weight", "enc_p.proj.weight", "enc_p.proj.bias",
        "dp.conv_1.weight", "dp.conv_1.bias", "dp.norm_1.gamma",
        "dp.norm_1.beta", "dp.conv_2.weight", "dp.conv_2.bias",
        "dp.norm_2.gamma", "dp.norm_2.beta", "dp.proj.weight", "dp.proj.bias",
    };
    for (const char* name : required) {
        if (!validate_tensor(name)) return false;
    }
    const ggml_tensor* duration_conv = tensor("dp.conv_1.weight");
    duration_hidden_channels_ = duration_conv->ne[2] > 1
                                    ? static_cast<int>(duration_conv->ne[2])
                                    : static_cast<int>(duration_conv->ne[1]);
    if (duration_hidden_channels_ <= 0 ||
        duration_hidden_channels_ > 2048) {
        std::fprintf(stderr,
                     "[V2Model] Invalid duration predictor width: %d\n",
                     duration_hidden_channels_);
        return false;
    }
    return true;
}

bool V2Model::load_flow_block(ModelLoader& loader, int flow_index) {
    if (!configure_architecture(loader) ||
        (flow_index != 0 && flow_index != 2 &&
         flow_index != 4 && flow_index != 6)) {
        return false;
    }
    loader_ = &loader;
    const std::string root =
        "flow.flows." + std::to_string(flow_index) + ".";
    return validate_tensor(root + "pre.weight") &&
           validate_tensor(root + "post.weight");
}

bool V2Model::load_decoder(ModelLoader& loader) {
    if (!configure_architecture(loader)) return false;
    loader_ = &loader;
    if (!validate_tensor("dec.conv_pre.weight") ||
        !validate_tensor("dec.conv_post.weight")) {
        return false;
    }
#if defined(INFLECT_LOW_MEMORY)
    if (!prepare_staged_decoder()) {
        return false;
    }
#else
    VocoderConfig config;
    config.sample_rate = sample_rate_;
    config.num_mels = latent_channels_;
    config.upsample_initial_channel = upsample_initial_channels_;
    config.activation = "leaky_relu";
    config.tensor_prefix = "dec";
    config.optional_biases = true;
    decoder_ = std::make_unique<VocoderModel>(config);
#endif
    return decoder_->load(loader);
}

#if defined(INFLECT_LOW_MEMORY)
bool V2Model::prepare_staged_decoder() {
    if (sample_rate_ <= 0 || latent_channels_ <= 0 ||
        upsample_initial_channels_ <= 0) {
        return false;
    }
    VocoderConfig config;
    config.sample_rate = sample_rate_;
    config.num_mels = latent_channels_;
    config.upsample_initial_channel =
        upsample_initial_channels_;
    config.activation = "leaky_relu";
    config.tensor_prefix = "dec";
    config.optional_biases = true;
    decoder_ = std::make_unique<VocoderModel>(config);
    return true;
}
#endif

bool V2Model::load(ModelLoader& loader) {
    if (!load_duration(loader)) return false;
    for (int flow : {0, 2, 4, 6}) {
        if (!load_flow_block(loader, flow)) return false;
    }
    return load_decoder(loader);
}

V2DurationFlowOutput V2Model::duration_and_flow(
    const std::vector<uint8_t>& tokens,
    float speed,
    float variation,
    uint64_t seed,
    const std::vector<float>* fixed_noise,
    bool run_flow
) const {
    V2DurationFlowOutput result;
    if (!loader_ || tokens.empty() || speed < 0.5f || speed > 2.0f ||
        variation < 0.0f || variation > 1.0f) {
        return result;
    }
    const int time = static_cast<int>(tokens.size());
    const ggml_tensor* embedding = tensor("enc_p.emb.weight");
    if (!embedding || embedding->ne[0] < hidden_channels_ ||
        embedding->ne[1] != 178) {
        return result;
    }
    std::vector<float> hidden(
        static_cast<size_t>(time) * hidden_channels_);
    TensorRow row;
    const float embedding_scale =
        std::sqrt(static_cast<float>(hidden_channels_));
    for (int t = 0; t < time; ++t) {
        const float* values =
            row.read(embedding, tokens[t], hidden_channels_);
        if (!values) return {};
        for (int c = 0; c < hidden_channels_; ++c) {
            hidden[static_cast<size_t>(t) * hidden_channels_ + c] =
                values[c] * embedding_scale;
        }
    }

    for (int layer = 0; layer < encoder_layers_; ++layer) {
        std::vector<float> residual;
        if (!attention(*loader_, layer, hidden, time, hidden_channels_,
                       attention_heads_, residual)) {
            return {};
        }
        add_in_place(hidden, residual);
        const std::string norm1 =
            "enc_p.encoder.norm_layers_1." + std::to_string(layer) + ".";
        if (!layer_norm(tensor(norm1 + "gamma"), tensor(norm1 + "beta"),
                        hidden, time, hidden_channels_)) {
            return {};
        }
        const std::string ffn =
            "enc_p.encoder.ffn_layers." + std::to_string(layer) + ".";
        std::vector<float> expanded;
        if (!conv1d(tensor(ffn + "conv_1.weight"), tensor(ffn + "conv_1.bias"),
                    hidden, time, hidden_channels_, filter_channels_,
                    3, 1, expanded)) {
            return {};
        }
        for (float& value : expanded) value = std::max(0.0f, value);
        if (!conv1d(tensor(ffn + "conv_2.weight"), tensor(ffn + "conv_2.bias"),
                    expanded, time, filter_channels_, hidden_channels_,
                    3, 1, residual)) {
            return {};
        }
        add_in_place(hidden, residual);
        const std::string norm2 =
            "enc_p.encoder.norm_layers_2." + std::to_string(layer) + ".";
        if (!layer_norm(tensor(norm2 + "gamma"), tensor(norm2 + "beta"),
                        hidden, time, hidden_channels_)) {
            return {};
        }
    }
    debug_time_major(
        "duration/encoder_hidden", hidden, time, hidden_channels_);
    std::vector<float> token_mask(time, 1.0f);
    debug_vector("duration/token_mask", token_mask,
                 "[1,1," + std::to_string(time) + "]");

    std::vector<float> stats;
    if (!conv1d(tensor("enc_p.proj.weight"), tensor("enc_p.proj.bias"),
                hidden, time, hidden_channels_, latent_channels_ * 2,
                1, 1, stats)) {
        return {};
    }
    if (!debug_directory().empty()) {
        std::vector<float> mean(
            static_cast<size_t>(time) * latent_channels_);
        std::vector<float> logs(mean.size());
        for (int t = 0; t < time; ++t) {
            std::copy_n(
                stats.data() +
                    static_cast<size_t>(t) * latent_channels_ * 2,
                latent_channels_,
                mean.data() + static_cast<size_t>(t) * latent_channels_);
            std::copy_n(
                stats.data() +
                    static_cast<size_t>(t) * latent_channels_ * 2 +
                    latent_channels_,
                latent_channels_,
                logs.data() + static_cast<size_t>(t) * latent_channels_);
        }
        debug_time_major(
            "duration/prior_mean", mean, time, latent_channels_);
        debug_time_major(
            "duration/prior_log_scale", logs, time, latent_channels_);
    }
    std::vector<float> duration_hidden;
    if (!conv1d(tensor("dp.conv_1.weight"), tensor("dp.conv_1.bias"),
                hidden, time, hidden_channels_, duration_hidden_channels_,
                3, 1, duration_hidden)) {
        return {};
    }
    for (float& value : duration_hidden) value = std::max(0.0f, value);
    if (!layer_norm(tensor("dp.norm_1.gamma"), tensor("dp.norm_1.beta"),
                    duration_hidden, time, duration_hidden_channels_)) {
        return {};
    }
    std::vector<float> duration_second;
    if (!conv1d(tensor("dp.conv_2.weight"), tensor("dp.conv_2.bias"),
                duration_hidden, time, duration_hidden_channels_,
                duration_hidden_channels_, 3, 1, duration_second)) {
        return {};
    }
    for (float& value : duration_second) value = std::max(0.0f, value);
    if (!layer_norm(tensor("dp.norm_2.gamma"), tensor("dp.norm_2.beta"),
                    duration_second, time, duration_hidden_channels_)) {
        return {};
    }
    std::vector<float> log_duration;
    if (!conv1d(tensor("dp.proj.weight"), tensor("dp.proj.bias"),
                duration_second, time, duration_hidden_channels_, 1, 1, 1,
                log_duration)) {
        return {};
    }
    debug_vector("duration/log_duration", log_duration,
                 "[1,1," + std::to_string(time) + "]");

    result.durations.resize(time);
    int64_t total_frames = 0;
    for (int t = 0; t < time; ++t) {
        const double raw = std::ceil(std::exp(static_cast<double>(log_duration[t])) /
                                     static_cast<double>(speed));
        const int32_t duration = static_cast<int32_t>(
            std::max(0.0, std::min(raw, 4000.0)));
        result.durations[t] = duration;
        total_frames += duration;
        if (total_frames > 4000) {
            std::fprintf(stderr, "[V2Model] Neural chunk exceeds 4000 latent frames\n");
            return {};
        }
    }
    if (!debug_directory().empty()) {
        debug_record("duration/integer_durations", "i32",
                     "[1,1," + std::to_string(time) + "]",
                     result.durations.size(), result.durations.data(),
                     result.durations.size() * sizeof(int32_t));
    }
    result.frames = std::max<int>(1, static_cast<int>(total_frames));
    result.channels = latent_channels_;
    result.latent.resize(
        static_cast<size_t>(result.frames) * latent_channels_);
    if (!debug_directory().empty()) {
        const int32_t length = result.frames;
        debug_record("duration/latent_lengths", "i32", "[1]", 1, &length,
                     sizeof(length));
        std::vector<float> latent_mask(result.frames, 1.0f);
        debug_vector("duration/latent_mask", latent_mask,
                     "[1,1," + std::to_string(result.frames) + "]");
    }
    DeterministicRNG random(seed);
    const size_t required_noise = result.latent.size();
    if (fixed_noise && fixed_noise->size() != required_noise) {
        std::fprintf(stderr,
                     "[V2Model] Fixed noise size mismatch expected=%zu got=%zu\n",
                     required_noise, fixed_noise->size());
        return {};
    }
    int frame = 0;
    std::vector<float> debug_noise;
    if (!debug_directory().empty()) debug_noise.resize(required_noise);
    std::vector<float> debug_expanded_mean;
    std::vector<float> debug_expanded_logs;
    if (!debug_directory().empty()) {
        debug_expanded_mean.resize(required_noise);
        debug_expanded_logs.resize(required_noise);
    }
    for (int token = 0; token < time; ++token) {
        for (int repeat = 0; repeat < result.durations[token]; ++repeat, ++frame) {
            for (int c = 0; c < latent_channels_; ++c) {
                const float mean =
                    stats[static_cast<size_t>(token) *
                              latent_channels_ * 2 +
                          c];
                const float log_scale =
                    stats[static_cast<size_t>(token) *
                              latent_channels_ * 2 +
                          latent_channels_ + c];
                const size_t index =
                    static_cast<size_t>(frame) * latent_channels_ + c;
                const float noise = fixed_noise ? (*fixed_noise)[index] : random.randn();
                if (!debug_noise.empty()) debug_noise[index] = noise;
                if (!debug_expanded_mean.empty()) {
                    debug_expanded_mean[index] = mean;
                    debug_expanded_logs[index] = log_scale;
                }
                result.latent[index] =
                    mean + noise * std::exp(log_scale) * variation;
            }
        }
    }
    if (frame == 0) {
        std::fill(result.latent.begin(), result.latent.end(), 0.0f);
    }
    if (!debug_noise.empty()) {
        debug_time_major("duration/expanded_mean", debug_expanded_mean,
                         result.frames, latent_channels_);
        debug_time_major("duration/expanded_log_scale", debug_expanded_logs,
                         result.frames, latent_channels_);
        debug_time_major(
            "flow/noise", debug_noise, result.frames, latent_channels_);
    }
    debug_time_major(
        "flow/latent_prior", result.latent, result.frames, latent_channels_);

    if (run_flow) {
        for (int flow = 6; flow >= 0; flow -= 2) {
            if (!reverse_flow_block(result, flow)) {
                return {};
            }
        }
    }
    return result;
}

bool V2Model::reverse_flow_block(V2DurationFlowOutput& output,
                                 int flow_index) const {
    if (!loader_ || output.frames <= 0 ||
        output.channels != latent_channels_ ||
        output.latent.size() !=
            static_cast<size_t>(output.frames) * latent_channels_ ||
        (flow_index != 0 && flow_index != 2 &&
         flow_index != 4 && flow_index != 6)) {
        return false;
    }
    flip_channels(output.latent, output.frames, latent_channels_);
    debug_time_major("flow/block_" + std::to_string(flow_index) + "/input",
                     output.latent, output.frames, latent_channels_);
    if (!reverse_coupling(*loader_, flow_index,
                          output.latent, output.frames,
                          latent_channels_, hidden_channels_)) {
        return false;
    }
    debug_time_major("flow/block_" + std::to_string(flow_index) + "/output",
                     output.latent, output.frames, latent_channels_);
    if (flow_index == 0) {
        debug_time_major("flow/latent", output.latent,
                         output.frames, latent_channels_);
    }
    return true;
}

void V2Model::decode_streaming(
    const std::vector<float>& latent,
    int frames,
    int chunk_frames,
    ggml_backend_t backend,
    AudioCallback callback
) {
    if (!decoder_ || frames <= 0 ||
        latent.size() != static_cast<size_t>(frames) * latent_channels_) {
        return;
    }
    decoder_->vocode_streaming(
        latent, latent_channels_, frames, chunk_frames,
        backend, std::move(callback));
}

#if defined(INFLECT_LOW_MEMORY)
bool V2Model::decode_staged(
    const std::string& model_path,
    const std::vector<float>& latent,
    int frames,
    ggml_backend_t backend,
    AudioCallback callback
) {
    if (!decoder_ || frames <= 0 ||
        latent.size() !=
            static_cast<size_t>(frames) *
                latent_channels_) {
        return false;
    }
    return decoder_->vocode_staged(
        model_path,
        latent,
        latent_channels_,
        frames,
        backend,
        std::move(callback));
}
#endif

} // namespace inflect
