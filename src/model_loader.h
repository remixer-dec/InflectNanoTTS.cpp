#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <gguf.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace inflect {

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();

    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    // Load GGUF file, allocate backend buffer for weights
    bool load(const std::string& path);

    // ── Tensor access ──────────────────────────────────────────────
    ggml_tensor* get_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

    // ── Metadata access ────────────────────────────────────────────
    int32_t     get_i32(const std::string& key, int32_t default_val = 0) const;
    float       get_f32(const std::string& key, float default_val = 0.0f) const;
    std::string get_string(const std::string& key, const std::string& default_val = "") const;

    // ── Contexts ───────────────────────────────────────────────────
    ggml_context*         ctx()     const { return ctx_; }
    ggml_backend_buffer_t buffer()  const { return buffer_; }

    // Debug
    std::vector<std::string> tensor_names() const;
    size_t n_tensors() const { return tensor_map_.size(); }

private:
    gguf_context*         gguf_   = nullptr;
    ggml_context*         ctx_    = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensor_map_;
};

} // namespace inflect
