#pragma once

#include "../ggml/include/ggml.h"
#include "../ggml/include/ggml-backend.h"
#include "../ggml/include/ggml-alloc.h"
#include "../ggml/include/gguf.h"
#include <cstdio>
#include <string>
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
    bool load_selected(const std::string& path, const std::vector<std::string>& prefixes);
    bool open(const std::string& path);
    bool select(const std::vector<std::string>& prefixes);
    void release_selected();
    // Optional bounded cache for small models; larger models stay stage-loaded.
    bool cache_weights(size_t max_bytes);

    // ── Tensor access ──────────────────────────────────────────────
    ggml_tensor* get_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

    // ── Metadata access ────────────────────────────────────────────
    int32_t     get_i32(const std::string& key, int32_t default_val = 0) const;
    float       get_f32(const std::string& key, float default_val = 0.0f) const;
    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    bool has_key(const std::string& key) const;

    // ── Contexts ───────────────────────────────────────────────────
    ggml_context*         ctx()     const { return ctx_; }
    ggml_backend_buffer_t buffer()  const { return buffer_; }

    // Debug
    std::vector<std::string> tensor_names() const;
    size_t n_tensors() const;

private:
    void release_weights();

    gguf_context*         gguf_   = nullptr;
    ggml_context*         ctx_    = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::FILE*            file_   = nullptr;
    std::string           path_;
    std::vector<ggml_tensor*> selected_tensors_;
    bool cached_weights_ = false;
};

} // namespace inflect
