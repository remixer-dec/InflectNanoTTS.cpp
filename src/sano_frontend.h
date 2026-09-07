#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace inflect {

struct SanoPiperConfig;

struct SanoFrontendResult {
    bool ok = false;
    std::string normalized_text;
    // Raw voice phoneme ids, without Piper framing/padding.
    std::vector<uint8_t> phoneme_ids;
    // Piper framing: [BOS, PAD, (id, PAD)*, EOS].
    std::vector<int32_t> tokens;
    std::vector<std::string> oov_words;
};

// Flash/file-backed Sano Piperlite frontend.  SNL1 is intentionally separate
// from IVL2: Inflect-v2 lexicons remain byte-for-byte compatible and can keep
// using V2Frontend.  SNL1 uses a fixed open-addressed hash table for expected
// O(1) word lookup instead of a sparse sorted scan.
class SanoFrontend {
public:
    SanoFrontend() = default;
    ~SanoFrontend();

    SanoFrontend(const SanoFrontend&) = delete;
    SanoFrontend& operator=(const SanoFrontend&) = delete;

    bool load_lexicon(const std::string& path, const SanoPiperConfig& config);
    bool loaded() const { return file_ != nullptr && entry_count_ != 0; }

    SanoFrontendResult process(
        const std::string& text,
        const SanoPiperConfig& config) const;

    bool lookup(std::string_view word, std::vector<uint8_t>& ids) const;
    std::string normalize(const std::string& text) const;

    uint32_t entry_count() const { return entry_count_; }
    uint32_t bucket_count() const { return bucket_count_; }
    const std::string& map_hash() const { return map_hash_; }
    const std::string& language() const { return language_; }

private:
    std::FILE* file_ = nullptr;
    std::string path_;
    std::string map_hash_;
    std::string language_;
    uint32_t entry_count_ = 0;
    uint32_t bucket_count_ = 0;
    uint32_t block_count_ = 0;
    uint64_t bucket_offset_ = 0;
    uint64_t hash_index_offset_ = 0;
    uint64_t block_offset_ = 0;
    uint64_t blob_offset_ = 0;
    uint64_t file_size_ = 0;
    uint16_t max_word_bytes_ = 0;
    uint16_t max_token_bytes_ = 0;
    bool snl2_ = false;
    mutable std::vector<char> word_scratch_;
    mutable std::vector<uint8_t> block_scratch_;

    bool spell_ascii_word(
        std::string_view word,
        const SanoPiperConfig& config,
        std::vector<uint8_t>& ids) const;
};

} // namespace inflect
