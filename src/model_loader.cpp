#include "model_loader.h"
#include "inflect-nano.h"
#include "memory_trace.h"
#include "../ggml/include/ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace inflect {

ModelLoader::~ModelLoader() {
    release_weights();
    if (file_)   std::fclose(file_);
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
    return open(path) && select(prefixes);
}

bool ModelLoader::open(const std::string& path) {
    if (gguf_) {
        if (path_ == path) return true;
        fprintf(stderr,
                "[ModelLoader] Loader already indexes a different GGUF: %s\n",
                path_.c_str());
        return false;
    }

    const uint32_t started_ms = runtime_now_ms();

    // Keep one parsed tensor catalog and one file handle across staged loads.
    // Only the selected backend weight buffer is replaced between stages.
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

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        if (!ggml_get_tensor(ctx_, name)) {
            fprintf(stderr,
                    "[ModelLoader] GGUF tensor missing from context: %s\n",
                    name);
            return false;
        }
    }

    file_ = std::fopen(path.c_str(), "rb");
    if (!file_) {
        fprintf(stderr, "[ModelLoader] Failed to reopen GGUF: %s\n", path.c_str());
        return false;
    }
    path_ = path;
    fprintf(stderr,
            "[ModelLoader] Indexed %d tensors elapsed_ms=%u from %s\n",
            n_tensors,
            static_cast<unsigned>(runtime_now_ms() - started_ms),
            path.c_str());
    return true;
}

void ModelLoader::release_weights() {
    for (ggml_tensor* tensor : selected_tensors_) {
        tensor->buffer = nullptr;
        tensor->data = nullptr;
    }
    selected_tensors_.clear();
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
}

void ModelLoader::release_selected() {
    release_weights();
}

bool ModelLoader::select(const std::vector<std::string>& prefixes) {
    if (!gguf_ || !ctx_ || !file_) {
        fprintf(stderr, "[ModelLoader] Cannot select tensors before opening GGUF\n");
        return false;
    }

    const uint32_t started_ms = runtime_now_ms();
    const std::vector<std::string>* selected_prefixes = prefixes.empty() ? nullptr : &prefixes;
    release_weights();

    struct SelectedTensor {
        ggml_tensor* tensor;
        size_t file_offset;
        size_t buffer_offset;
        size_t bytes;
    };

    const int n_tensors = gguf_get_n_tensors(gguf_);
    ggml_backend_buffer_type_t buft = runtime_weight_buffer_type();
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t total_alloc = 0;
    size_t total_size = 0;
    std::vector<SelectedTensor> selected;
    selected.reserve(n_tensors);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        if (!selected_tensor(name, selected_prefixes)) continue;
        ggml_tensor* tensor = ggml_get_tensor(ctx_, name);
        total_alloc = align_up(total_alloc, alignment);
        const size_t bytes = ggml_nbytes(tensor);
        selected.push_back({
            tensor,
            gguf_get_data_offset(gguf_) + gguf_get_tensor_offset(gguf_, i),
            total_alloc,
            bytes,
        });
        total_alloc += ggml_backend_buft_get_alloc_size(buft, tensor);
        total_size += bytes;
    }
    if (selected.empty()) {
        fprintf(stderr, "[ModelLoader] Tensor selection is empty in %s\n",
                path_.c_str());
        return false;
    }

    buffer_ = ggml_backend_buft_alloc_buffer(buft, total_alloc);
    if (!buffer_) {
        fprintf(stderr, "[ModelLoader] Failed to allocate backend tensor buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    mem_trace_rss("loader buffer allocated");

    char* base = static_cast<char*>(ggml_backend_buffer_get_base(buffer_));
    for (const SelectedTensor& item : selected) {
        if (ggml_backend_tensor_alloc(
                buffer_, item.tensor, base + item.buffer_offset) !=
            GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[ModelLoader] Failed to allocate tensor %s\n",
                    item.tensor->name);
            return false;
        }
        selected_tensors_.push_back(item.tensor);
    }
    const uint32_t allocated_ms = runtime_now_ms();

    // GGUF writes tensors in aligned file order. When the selected tensors
    // form one packed range, their file offsets exactly match our backend
    // offsets, so one seek/read replaces per-tensor seeks and copies.
    const size_t first_file_offset = selected.front().file_offset;
    bool bulk_layout = ggml_backend_buffer_is_host(buffer_);
    for (const SelectedTensor& item : selected) {
        bulk_layout =
            bulk_layout &&
            item.file_offset == first_file_offset + item.buffer_offset;
    }

    if (bulk_layout) {
        if (std::fseek(file_, static_cast<long>(first_file_offset), SEEK_SET) != 0) {
            fprintf(stderr, "[ModelLoader] Failed to seek bulk tensor range\n");
            return false;
        }
        size_t done = 0;
        while (done < total_alloc) {
            const size_t chunk =
                std::min<size_t>(64 * 1024, total_alloc - done);
            const size_t read =
                std::fread(base + done, 1, chunk, file_);
            if (read != chunk) {
                fprintf(stderr,
                        "[ModelLoader] Short bulk read: %zu/%zu\n",
                        done + read, total_alloc);
                return false;
            }
            done += chunk;
        }
    } else {
        std::vector<uint8_t> data(8 * 1024);
        for (const SelectedTensor& item : selected) {
            if (std::fseek(
                    file_, static_cast<long>(item.file_offset), SEEK_SET) != 0) {
                fprintf(stderr, "[ModelLoader] Failed to seek tensor %s\n",
                        item.tensor->name);
                return false;
            }
            size_t done = 0;
            while (done < item.bytes) {
                const size_t chunk =
                    std::min(data.size(), item.bytes - done);
                const size_t read =
                    std::fread(data.data(), 1, chunk, file_);
                if (read != chunk) {
                    fprintf(
                        stderr,
                        "[ModelLoader] Short read for tensor %s: %zu/%zu\n",
                        item.tensor->name, done + read, item.bytes);
                    return false;
                }
                ggml_backend_tensor_set(
                    item.tensor, data.data(), done, chunk);
                done += chunk;
            }
        }
    }
    mem_release_to_os();
    mem_trace_rss("loader tensors loaded");

    fprintf(
        stderr,
        "[ModelLoader] Loaded %u/%d tensors (%.2f MB) io=%s "
        "alloc_ms=%u read_ms=%u total_ms=%u from %s\n",
        static_cast<unsigned>(selected.size()),
        n_tensors,
        total_size / 1024.0 / 1024.0,
        bulk_layout ? "bulk" : "tensor",
        static_cast<unsigned>(allocated_ms - started_ms),
        static_cast<unsigned>(runtime_now_ms() - allocated_ms),
        static_cast<unsigned>(runtime_now_ms() - started_ms),
        path_.c_str());
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

bool ModelLoader::has_key(const std::string& key) const {
    return gguf_ && gguf_find_key(gguf_, key.c_str()) >= 0;
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
