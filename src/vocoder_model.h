#pragma once

#include "model_config.h"
#include "model_loader.h"
#include "../ggml/include/ggml.h"
#include "../ggml/include/ggml-backend.h"
#include <memory>
#include <string>
#include <vector>

namespace inflect {

struct ResBlockWeights {
    std::vector<ggml_tensor*> convs1_w;
    std::vector<ggml_tensor*> convs1_b;
    std::vector<ggml_tensor*> convs2_w;
    std::vector<ggml_tensor*> convs2_b;
    std::vector<ggml_tensor*> acts1_alpha;
    std::vector<ggml_tensor*> acts2_alpha;
};

struct VocoderWeights {
    ggml_tensor* conv_pre_w = nullptr;
    ggml_tensor* conv_pre_b = nullptr;
    std::vector<ggml_tensor*> ups_w;
    std::vector<ggml_tensor*> ups_b;
    std::vector<ggml_tensor*> up_acts_alpha;
    std::vector<ResBlockWeights> resblocks;
    ggml_tensor* post_act_alpha = nullptr;
    ggml_tensor* conv_post_w = nullptr;
    ggml_tensor* conv_post_b = nullptr;
};

struct QuantConvTranspose1dOpData {
    const char* profile_label;
    int kernel_size;
    int stride;
    int crop_left;
};

struct VocoderQuantConv1dOpData {
    const char* profile_label;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
};

class VocoderModel {
public:
    explicit VocoderModel(const VocoderConfig& config);
    ~VocoderModel();

    bool load(ModelLoader& loader);

    std::vector<float> vocode(
        const std::vector<float>& mel,
        int n_mels,
        int n_frames,
        ggml_backend_t backend
    );

    void vocode_streaming(
        const std::vector<float>& mel,
        int n_mels,
        int n_frames,
        int chunk_frames,
        ggml_backend_t backend,
        AudioCallback callback
    );

#if defined(INFLECT_LOW_MEMORY)
    bool vocode_staged(
        const std::string& model_path,
        const std::vector<float>& mel,
        int n_mels,
        int n_frames,
        ggml_backend_t backend,
        AudioCallback callback
    );
#endif

    int total_upsample() const;
    const VocoderConfig& config() const { return config_; }

private:
    VocoderConfig config_;
    VocoderWeights weights_;
    ggml_context* wctx_ = nullptr; // Non-owning; ModelLoader owns the context.
    std::vector<VocoderQuantConv1dOpData> quant_conv1d_ops_;
    std::vector<QuantConvTranspose1dOpData> quant_conv_transpose_ops_;

    void fold_weight_norm(
        ggml_tensor* dst,
        ggml_tensor* weight_v,
        ggml_tensor* weight_g,
        int dim0,
        int dim1,
        int dim2,
        bool is_transpose
    );

    ggml_tensor* snake(
        ggml_context* gctx,
        ggml_tensor* x,
        ggml_tensor* log_alpha
    );

    ggml_tensor* build_resblock(
        ggml_context* gctx,
        ggml_tensor* x,
        const ResBlockWeights& w,
        int kernel_size,
        int max_depth
    );

    ggml_cgraph* build_vocoder_graph(
        ggml_context* gctx,
        ggml_tensor* mel
    );

#if defined(INFLECT_LOW_MEMORY)
    bool load_pre_stage(ModelLoader& loader);
    bool load_upsample_stage(ModelLoader& loader, int stage);
    bool load_post_stage(ModelLoader& loader);

    std::vector<float> run_pre_stage(
        const std::vector<float>& mel,
        int n_mels,
        int n_frames,
        ggml_backend_t backend
    );
    std::vector<float> run_upsample_stage(
        std::vector<float>& input,
        int input_frames,
        int stage,
        ggml_backend_t backend
    );
    std::vector<float> run_upsample_stage_once(
        std::vector<float>& input,
        int input_frames,
        int stage,
        ggml_backend_t backend,
        bool log_completion
    );
    std::vector<float> run_post_stage(
        std::vector<float>& input,
        int n_frames,
        ggml_backend_t backend
    );
#endif
};

} // namespace inflect
