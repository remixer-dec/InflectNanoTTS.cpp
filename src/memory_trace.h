#pragma once

#include <ggml.h>
#include <ggml-alloc.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__linux__)
#include <malloc.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#endif

namespace inflect {

inline bool mem_trace_enabled() {
    const char* env = std::getenv("INFLECT_MEM_TRACE");
    return env && std::strcmp(env, "1") == 0;
}

inline bool current_rss_kb(size_t& rss_kb) {
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string key;
    while (f >> key) {
        if (key == "VmRSS:") {
            f >> rss_kb;
            return true;
        }
        std::string rest;
        std::getline(f, rest);
    }
    return false;
#elif defined(__APPLE__) && defined(__MACH__)
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return false;
    }
    rss_kb = static_cast<size_t>(info.resident_size / 1024);
    return true;
#else
    (void)rss_kb;
    return false;
#endif
}

inline void mem_trace_rss(const char* label) {
    if (!mem_trace_enabled()) return;
    size_t rss_kb = 0;
    if (current_rss_kb(rss_kb)) {
        std::fprintf(stderr, "[mem] %s rss=%zu KB (%.2f MB)\n",
                     label, rss_kb, rss_kb / 1024.0);
    }
}

inline void mem_release_to_os() {
#if defined(__linux__)
    malloc_trim(0);
#endif
}

inline void mem_trace_graph(const char* label, const ggml_context* gctx, ggml_gallocr_t allocr) {
    if (!mem_trace_enabled()) return;
    const size_t meta_bytes = ggml_used_mem(gctx);
    const size_t alloc_bytes = ggml_gallocr_get_buffer_size(allocr, 0);
    std::fprintf(stderr, "[mem] %s graph_meta=%zu bytes (%.2f MB) graph_buffer=%zu bytes (%.2f MB)\n",
                 label,
                 meta_bytes, meta_bytes / (1024.0 * 1024.0),
                 alloc_bytes, alloc_bytes / (1024.0 * 1024.0));
}

} // namespace inflect
