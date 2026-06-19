#include "model_loader.h"
#include <ggml-cpu.h>
#include <cstdio>
#include <cstring>
#include <vector>

namespace inflect {

ModelLoader::~ModelLoader() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_)    ggml_free(ctx_);
    if (gguf_)   gguf_free(gguf_);
}

bool ModelLoader::load(const std::string& path) {
    // ── 1. Open GGUF and let GGML create tensor metadata ───────────
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_,
    };
    gguf_ = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_) {
        fprintf(stderr, "[ModelLoader] Failed to open GGUF: %s\n", path.c_str());
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(gguf_);
    if (!ctx_) {
        fprintf(stderr, "[ModelLoader] GGUF did not create a ggml_context\n");
        return false;
    }

    // ── 2. Index tensors by name ───────────────────────────────────
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* tensor = ggml_get_tensor(ctx_, name);
        if (!tensor) {
            fprintf(stderr, "[ModelLoader] GGUF tensor missing from context: %s\n", name);
            return false;
        }
        tensor_map_[name] = tensor;
    }

    // ── 3. Allocate CPU backend storage for all weight tensors ─────
    buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, ggml_backend_cpu_buffer_type());
    if (!buffer_) {
        fprintf(stderr, "[ModelLoader] Failed to allocate backend tensor buffer\n");
        return false;
    }

    // ── 4. Load tensor data from GGUF into backend tensors ─────────
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[ModelLoader] Failed to reopen GGUF: %s\n", path.c_str());
        return false;
    }

    size_t total_size = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* tensor = tensor_map_[name];

        size_t offset = gguf_get_data_offset(gguf_) + gguf_get_tensor_offset(gguf_, i);
        size_t nbytes = ggml_nbytes(tensor);
        total_size += nbytes;

        std::vector<uint8_t> data(nbytes);
        if (fseek(f, (long)offset, SEEK_SET) != 0) {
            fprintf(stderr, "[ModelLoader] Failed to seek tensor %s\n", name);
            fclose(f);
            return false;
        }

        size_t read = fread(data.data(), 1, nbytes, f);

        if (read != nbytes) {
            fprintf(stderr, "[ModelLoader] Short read for tensor %s: %zu/%zu\n", name, read, nbytes);
            fclose(f);
            return false;
        }

        ggml_backend_tensor_set(tensor, data.data(), 0, nbytes);
    }
    fclose(f);

    fprintf(stderr, "[ModelLoader] Loaded %d tensors (%.2f MB) from %s\n",
            n_tensors, total_size / 1024.0 / 1024.0, path.c_str());
    return true;
}

ggml_tensor* ModelLoader::get_tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
        fprintf(stderr, "[ModelLoader] Tensor not found: %s\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

bool ModelLoader::has_tensor(const std::string& name) const {
    return tensor_map_.find(name) != tensor_map_.end();
}

int32_t ModelLoader::get_i32(const std::string& key, int32_t default_val) const {
    int idx = gguf_find_key(gguf_, key.c_str());
    if (idx < 0) return default_val;
    switch (gguf_get_kv_type(gguf_, idx)) {
        case GGUF_TYPE_UINT8:  return (int32_t)gguf_get_val_u8(gguf_, idx);
        case GGUF_TYPE_INT8:   return (int32_t)gguf_get_val_i8(gguf_, idx);
        case GGUF_TYPE_UINT16: return (int32_t)gguf_get_val_u16(gguf_, idx);
        case GGUF_TYPE_INT16:  return (int32_t)gguf_get_val_i16(gguf_, idx);
        case GGUF_TYPE_UINT32: return (int32_t)gguf_get_val_u32(gguf_, idx);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(gguf_, idx);
        case GGUF_TYPE_UINT64: return (int32_t)gguf_get_val_u64(gguf_, idx);
        case GGUF_TYPE_INT64:  return (int32_t)gguf_get_val_i64(gguf_, idx);
        default: return default_val;
    }
}

float ModelLoader::get_f32(const std::string& key, float default_val) const {
    int idx = gguf_find_key(gguf_, key.c_str());
    if (idx < 0) return default_val;
    switch (gguf_get_kv_type(gguf_, idx)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf_, idx);
        case GGUF_TYPE_FLOAT64: return (float)gguf_get_val_f64(gguf_, idx);
        case GGUF_TYPE_UINT8:   return (float)gguf_get_val_u8(gguf_, idx);
        case GGUF_TYPE_INT8:    return (float)gguf_get_val_i8(gguf_, idx);
        case GGUF_TYPE_UINT16:  return (float)gguf_get_val_u16(gguf_, idx);
        case GGUF_TYPE_INT16:   return (float)gguf_get_val_i16(gguf_, idx);
        case GGUF_TYPE_UINT32:  return (float)gguf_get_val_u32(gguf_, idx);
        case GGUF_TYPE_INT32:   return (float)gguf_get_val_i32(gguf_, idx);
        case GGUF_TYPE_UINT64:  return (float)gguf_get_val_u64(gguf_, idx);
        case GGUF_TYPE_INT64:   return (float)gguf_get_val_i64(gguf_, idx);
        default: return default_val;
    }
}

std::string ModelLoader::get_string(const std::string& key, const std::string& default_val) const {
    int idx = gguf_find_key(gguf_, key.c_str());
    if (idx < 0) return default_val;
    if (gguf_get_kv_type(gguf_, idx) != GGUF_TYPE_STRING) return default_val;
    const char* s = gguf_get_val_str(gguf_, idx);
    return s ? std::string(s) : default_val;
}

std::vector<std::string> ModelLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_map_.size());
    for (const auto& [k, _] : tensor_map_) {
        names.push_back(k);
    }
    return names;
}

} // namespace inflect
