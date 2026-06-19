#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

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
    // Binary dictionary: word → list of (phoneme_id, tone)
    struct DictEntry {
        std::vector<std::pair<uint8_t, uint8_t>> phonemes; // (phone_id, tone)
    };
    std::unordered_map<std::string, DictEntry> dict_;

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
