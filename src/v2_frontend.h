#pragma once

#include "v2_symbols.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace inflect {

struct V2FrontendResult {
    std::string raw_text;
    std::string normalized_text;
    std::vector<uint8_t> unblanked_tokens;
    std::vector<uint8_t> blanked_tokens;
    std::vector<std::string> oov_words;
};

// Flash/file-backed Inflect v2 frontend. It intentionally does not share any
// dictionary or normalization state with the v1 TinyTTS frontend.
class V2Frontend {
public:
    V2Frontend() = default;
    ~V2Frontend();

    V2Frontend(const V2Frontend&) = delete;
    V2Frontend& operator=(const V2Frontend&) = delete;

    bool load_lexicon(const std::string& bin_path,
                      const std::string& index_path = "");
    bool loaded() const { return lexicon_.is_open() && entry_count_ != 0; }

    V2FrontendResult process(const std::string& text) const;
    std::string normalize(const std::string& text) const;
    bool lookup(std::string_view word, std::vector<uint8_t>& tokens) const;

    uint32_t entry_count() const { return entry_count_; }
    uint32_t oov_count() const { return oov_count_; }
    const std::string& manifest_hash() const { return manifest_hash_; }

private:
    struct SparseEntry {
        uint32_t entry_index = 0;
        uint64_t offset = 0;
        std::string word;
    };

    mutable std::ifstream lexicon_;
    std::string lexicon_path_;
    uint32_t entry_count_ = 0;
    uint32_t sparse_stride_ = 0;
    uint64_t entries_offset_ = 0;
    uint64_t index_offset_ = 0;
    std::vector<SparseEntry> sparse_;
    mutable uint32_t oov_count_ = 0;
    std::string manifest_hash_;

    bool load_sidecar(const std::string& path);
    bool load_embedded_sparse_index();
    bool lookup_morphology(std::string_view word,
                           std::vector<uint8_t>& tokens) const;
    bool spell_word(std::string_view word, std::vector<uint8_t>& tokens) const;
};

} // namespace inflect
