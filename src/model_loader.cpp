#include "model_loader.h"
#include "inflect-nano.h"
#include "memory_trace.h"
#include <ggml-cpu.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace inflect {

ModelLoader::~ModelLoader() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_)    ggml_free(ctx_);
    if (gguf_)   gguf_free(gguf_);
}

static bool selected_tensor(const char* name, const std::vector<std::string>* prefixes) {
    if (!prefixes) return true;
    for (const std::string& prefix : *prefixes) {
        if (std::strncmp(name, prefix.c_str(), prefix.size()) == 0) {
            return true;
        }
    }
    return false;
}

static size_t align_up(size_t value, size_t align) {
    return align > 1 ? ((value + align - 1) / align) * align : value;
}

bool ModelLoader::load(const std::string& path) {
    return load_selected(path, {});
}

bool ModelLoader::load_selected(const std::string& path, const std::vector<std::string>& prefixes) {
    const std::vector<std::string>* selected_prefixes = prefixes.empty() ? nullptr : &prefixes;

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
    mem_trace_rss("loader gguf init");

    const int n_tensors = gguf_get_n_tensors(gguf_);
    if (!ctx_) {
        fprintf(stderr, "[ModelLoader] GGUF did not create a ggml_context\n");
        return false;
    }

    // ── 2. Verify tensors by name ──────────────────────────────────
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* tensor = ggml_get_tensor(ctx_, name);
        if (!tensor) {
            fprintf(stderr, "[ModelLoader] GGUF tensor missing from context: %s\n", name);
            return false;
        }
    }

    // ── 3. Allocate CPU backend storage for selected weight tensors ─
    ggml_backend_buffer_type_t buft = runtime_weight_buffer_type();
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t total_alloc = 0;
    int selected_count = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        if (!selected_tensor(name, selected_prefixes)) continue;
        ggml_tensor* tensor = ggml_get_tensor(ctx_, name);
        total_alloc = align_up(total_alloc, alignment);
        total_alloc += ggml_backend_buft_get_alloc_size(buft, tensor);
        selected_count++;
    }

    buffer_ = ggml_backend_buft_alloc_buffer(buft, total_alloc);
    if (!buffer_) {
        fprintf(stderr, "[ModelLoader] Failed to allocate backend tensor buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    mem_trace_rss("loader buffer allocated");

    char* base = static_cast<char*>(ggml_backend_buffer_get_base(buffer_));
    size_t cursor = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        if (!selected_tensor(name, selected_prefixes)) continue;
        ggml_tensor* tensor = ggml_get_tensor(ctx_, name);
        cursor = align_up(cursor, alignment);
        if (ggml_backend_tensor_alloc(buffer_, tensor, base + cursor) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[ModelLoader] Failed to allocate tensor %s\n", name);
            return false;
        }
        cursor += ggml_backend_buft_get_alloc_size(buft, tensor);
    }

    // ── 4. Load tensor data from GGUF into backend tensors ─────────
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[ModelLoader] Failed to reopen GGUF: %s\n", path.c_str());
        return false;
    }

    size_t total_size = 0;
    std::vector<uint8_t> data(8 * 1024);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        if (!selected_tensor(name, selected_prefixes)) continue;
        ggml_tensor* tensor = ggml_get_tensor(ctx_, name);

        size_t offset = gguf_get_data_offset(gguf_) + gguf_get_tensor_offset(gguf_, i);
        size_t nbytes = ggml_nbytes(tensor);
        total_size += nbytes;

        if (fseek(f, (long)offset, SEEK_SET) != 0) {
            fprintf(stderr, "[ModelLoader] Failed to seek tensor %s\n", name);
            fclose(f);
            return false;
        }

        size_t done = 0;
        while (done < nbytes) {
            const size_t chunk = std::min(data.size(), nbytes - done);
            size_t read = fread(data.data(), 1, chunk, f);
            if (read != chunk) {
                fprintf(stderr, "[ModelLoader] Short read for tensor %s: %zu/%zu\n",
                        name, done + read, nbytes);
                fclose(f);
                return false;
            }
            ggml_backend_tensor_set(tensor, data.data(), done, chunk);
            done += chunk;
        }
    }
    fclose(f);
    mem_release_to_os();
    mem_trace_rss("loader tensors loaded");

    fprintf(stderr, "[ModelLoader] Loaded %d/%d tensors (%.2f MB) from %s\n",
            selected_count, n_tensors, total_size / 1024.0 / 1024.0, path.c_str());
    return true;
}

ggml_tensor* ModelLoader::get_tensor(const std::string& name) const {
    ggml_tensor* tensor = ctx_ ? ggml_get_tensor(ctx_, name.c_str()) : nullptr;
    if (!tensor || !tensor->data) {
        fprintf(stderr, "[ModelLoader] Tensor not found: %s\n", name.c_str());
        return nullptr;
    }
    return tensor;
}

bool ModelLoader::has_tensor(const std::string& name) const {
    return ctx_ && ggml_get_tensor(ctx_, name.c_str()) != nullptr;
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
    const int n_tensors = gguf_ ? gguf_get_n_tensors(gguf_) : 0;
    names.reserve(n_tensors);
    for (int i = 0; i < n_tensors; i++) {
        names.emplace_back(gguf_get_tensor_name(gguf_, i));
    }
    return names;
}

size_t ModelLoader::n_tensors() const {
    return gguf_ ? (size_t)gguf_get_n_tensors(gguf_) : 0;
}

} // namespace inflect
