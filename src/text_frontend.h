#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>

namespace inflect {

struct PhonemeResult {
    std::vector<int32_t> phone_ids;
    std::vector<int32_t> tone_ids;
    std::vector<int32_t> lang_ids;
};

class TextFrontend {
public:
    TextFrontend();
    ~TextFrontend();

    // Load binary cmudict (compiled by compile_cmudict.py)
    bool load_cmudict(const std::string& path);

    // Full pipeline: raw text → phoneme/tone/lang IDs
    PhonemeResult process(const std::string& text);

    // Individual steps (exposed for testing)
    std::string clean_text(const std::string& text);
    std::string normalize_text(const std::string& text);

    // G2P for a single word
    std::vector<std::pair<std::string, int>> word_to_phonemes(
        const std::string& word
    ); // returns [(phoneme, tone), ...]

    // Phoneme symbol → ID
    int phoneme_to_id(const std::string& phoneme) const;

    // Insert blanks between tokens (ADD_BLANK pattern)
    static void insert_blanks(std::vector<int32_t>& ids, int32_t blank_id = 0);

    // Language ID for English
    static constexpr int32_t LANG_EN = 2;
    static constexpr int32_t TONE_OFFSET_EN = 7; // num_zh_tones + num_ja_tones

private:
    struct PhoneTone {
        int32_t phone_id = 0;
        int32_t tone = 0;
    };

    // Compact binary dictionary loaded from sorted cmudict.bin.
    struct DictEntry {
        uint32_t word_offset = 0;
        uint32_t phone_offset = 0;
        uint8_t word_len = 0;
        uint8_t phone_count = 0;
    };
    std::vector<DictEntry> dict_;
    std::string dict_words_;
    std::vector<std::pair<uint8_t, uint8_t>> dict_phones_;

    const DictEntry* find_dict_entry(std::string_view word) const;
    std::string_view dict_word(const DictEntry& entry) const;

    struct SparseIndexEntry {
        uint32_t word_offset = 0;
        uint8_t word_len = 0;
        uint64_t offset = 0;
        uint32_t entry_index = 0;
    };
    std::string dict_path_;
    std::vector<SparseIndexEntry> sparse_index_;
    std::string sparse_words_;
    mutable std::ifstream flash_dict_;
    uint32_t dict_count_ = 0;
    bool flash_cmu_ = false;
    bool legacy_phone_ids_ = false;

    bool load_cmudict_sparse_index(std::ifstream& f, uint32_t count);
    std::string_view sparse_word(const SparseIndexEntry& entry) const;
    bool lookup_flash_entry(
        std::string_view word,
        std::vector<PhoneTone>& phonemes
    ) const;
    std::vector<PhoneTone> word_to_phone_ids(const std::string& word);

    // Phoneme symbol → ID mapping (from symbols.py)
    std::unordered_map<std::string, int> phoneme_vocab_;
    void init_phoneme_vocab();

    // Number normalization (minimal C++ implementation)
    std::string normalize_numbers(const std::string& text);

    // Abbreviation expansion
    std::string expand_abbreviations(const std::string& text);

    // OOV fallback: spell out letter by letter
    std::vector<std::pair<std::string, int>> spell_out(const std::string& word);
};

} // namespace inflect
