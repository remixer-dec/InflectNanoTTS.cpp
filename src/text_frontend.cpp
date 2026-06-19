#include "text_frontend.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <regex>
#include <cstdlib>

namespace inflect {

static uint8_t remap_legacy_phone_id(uint8_t id) {
    // Older cmudict.bin files were built with an English-only symbol table.
    // Map those IDs into the full TinyTTS model vocabulary.
    static const uint8_t map[] = {
        0,   // _
        14,  // V
        21, 22, 23, 27, 28, 29, 30, 33, 34, 35, 39, 43, 44,
        45, 46, 49, 59, 65, 67, 68, 70, 71, 73, 74, 80, 81,
        82, 85, 87, 88, 89, 90, 99, 103, 108, 110, 111, 112,
        208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
    };
    return id < sizeof(map) ? map[id] : id;
}

// Phoneme vocabulary (from symbols.py)
// pad + sorted(all_symbols) + punctuation
// This is initialized to match the Python symbols list exactly.

TextFrontend::TextFrontend() {
    init_phoneme_vocab();
}

TextFrontend::~TextFrontend() = default;

void TextFrontend::init_phoneme_vocab() {
    // English and punctuation IDs from the full TinyTTS symbols.py table.
    phoneme_vocab_ = {
        {"_", 0},
        {"V", 14},
        {"aa", 21}, {"ae", 22}, {"ah", 23}, {"ao", 27}, {"aw", 28}, {"ay", 29},
        {"b", 30}, {"ch", 33}, {"d", 34}, {"dh", 35}, {"eh", 39}, {"er", 43},
        {"ey", 44}, {"f", 45}, {"g", 46}, {"hh", 49}, {"ih", 59}, {"iy", 65},
        {"jh", 67}, {"k", 68}, {"l", 70}, {"m", 71}, {"n", 73}, {"ng", 74},
        {"ow", 80}, {"oy", 81}, {"p", 82}, {"r", 85}, {"s", 87}, {"sh", 88},
        {"t", 89}, {"th", 90}, {"uh", 99}, {"uw", 103}, {"w", 108}, {"y", 110},
        {"z", 111}, {"zh", 112},
        {"!", 208}, {"?", 209}, {"…", 210}, {",", 211}, {".", 212}, {"'", 213},
        {"-", 214}, {"¿", 215}, {"¡", 216}, {"SP", 217}, {"UNK", 218},
    };
}

bool TextFrontend::load_cmudict(const std::string& path) {
    // Load the binary compiled cmudict
    // Current format: ["CMD2"][count: u32] repeated [word_len: u8][word]
    // [n_phones: u8][phone_id: u8, tone: u8] x n_phones.
    // Legacy files omit the magic and use English-only phone IDs.
    //
    // On MCU, this file would be memory-mapped from flash.

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "[TextFrontend] Failed to open cmudict: %s\n", path.c_str());
        return false;
    }

    uint32_t marker = 0;
    f.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (f.gcount() != sizeof(marker)) {
        fprintf(stderr, "[TextFrontend] Invalid cmudict header: %s\n", path.c_str());
        return false;
    }

    constexpr uint32_t kMagic = 0x32444d43; // "CMD2", little-endian
    uint32_t count = 0;
    bool legacy_phone_ids = false;
    if (marker == kMagic) {
        f.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (f.gcount() != sizeof(count)) {
            fprintf(stderr, "[TextFrontend] Invalid cmudict count: %s\n", path.c_str());
            return false;
        }
    } else {
        count = marker;
        legacy_phone_ids = true;
    }

    for (uint32_t entry_idx = 0; entry_idx < count && f.good(); entry_idx++) {
        uint8_t word_len = 0;
        f.read(reinterpret_cast<char*>(&word_len), 1);
        if (f.gcount() < 1) break;

        std::string word(word_len, '\0');
        f.read(&word[0], word_len);

        uint8_t n_phones;
        f.read(reinterpret_cast<char*>(&n_phones), 1);

        DictEntry entry;
        entry.phonemes.resize(n_phones);
        for (int i = 0; i < n_phones; i++) {
            f.read(reinterpret_cast<char*>(&entry.phonemes[i].first), 1);  // phone_id
            f.read(reinterpret_cast<char*>(&entry.phonemes[i].second), 1); // tone
        }

        dict_[word] = std::move(entry);
    }

    if (legacy_phone_ids) {
        for (auto& [word, entry] : dict_) {
            (void)word;
            for (auto& [phone_id, tone] : entry.phonemes) {
                phone_id = remap_legacy_phone_id(phone_id);
            }
        }
        fprintf(stderr, "[TextFrontend] Remapped legacy cmudict phone IDs to model vocabulary\n");
    }

    fprintf(stderr, "[TextFrontend] Loaded %zu dictionary entries\n", dict_.size());
    return true;
}

std::string TextFrontend::clean_text(const std::string& text) {
    // Port of clean_tinytts_text from text_cleaning.py
    std::string result = text;

    // Quote translation
    auto replace = [&](const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = result.find(from, pos)) != std::string::npos) {
            result.replace(pos, from.length(), to);
            pos += to.length();
        }
    };

    replace("\u2018", "'");
    replace("\u2019", "'");
    replace("\u201c", "");
    replace("\u201d", "");
    replace("\u2014", ",");
    replace("\u2013", ",");
    replace(";", ",");
    replace(":", ",");
    replace("\n", ".");
    replace("...", "\u2026");

    result = std::regex_replace(result, std::regex("\\s+"), " ");
    result = std::regex_replace(result, std::regex("\\s+([,.!?…])"), "$1");
    result = std::regex_replace(result, std::regex("([,.!?…]){2,}"), "$1");

    return result;
}

std::string TextFrontend::normalize_text(const std::string& text) {
    std::string result = text;
    // Lowercase
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    // Expand abbreviations
    result = expand_abbreviations(result);
    // Normalize numbers
    result = normalize_numbers(result);
    return result;
}

std::string TextFrontend::expand_abbreviations(const std::string& text) {
    // Match Python's abbreviation expansion: regex with word boundaries
    std::string result = text;
    static const std::vector<std::pair<std::string, std::string>> abbrevs = {
        {"mrs\\.", "misess"}, {"mr\\.", "mister"}, {"dr\\.", "doctor"},
        {"st\\.", "saint"}, {"co\\.", "company"}, {"jr\\.", "junior"},
        {"maj\\.", "major"}, {"gen\\.", "general"}, {"drs\\.", "doctors"},
        {"rev\\.", "reverend"}, {"lt\\.", "lieutenant"}, {"hon\\.", "honorable"},
        {"sgt\\.", "sergeant"}, {"capt\\.", "captain"}, {"esq\\.", "esquire"},
        {"ltd\\.", "limited"}, {"col\\.", "colonel"}, {"ft\\.", "fort"},
    };
    for (const auto& [pattern, expansion] : abbrevs) {
        std::regex re("\\b" + pattern, std::regex::icase);
        result = std::regex_replace(result, re, expansion);
    }
    return result;
}

std::string TextFrontend::normalize_numbers(const std::string& text) {
    // Minimal number normalization
    // Full implementation would port number_norm.py from the Python code
    // For now, this is a stub that handles basic cases
    return text;
}

std::vector<std::pair<std::string, int>> TextFrontend::word_to_phonemes(
    const std::string& word
) {
    std::vector<std::pair<std::string, int>> result;

    if (word.size() == 1 && phoneme_vocab_.find(word) != phoneme_vocab_.end()) {
        result.push_back({word, 0});
        return result;
    }

    // Look up in dictionary
    std::string upper_word = word;
    std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    auto it = dict_.find(upper_word);
    if (it != dict_.end()) {
        for (const auto& [phone_id, tone] : it->second.phonemes) {
            // Convert phone_id back to symbol (reverse lookup)
            // For now, return placeholder
            result.push_back({"phone_" + std::to_string(phone_id), tone});
        }
        return result;
    }

    // OOV: spell out
    return spell_out(word);
}

std::vector<std::pair<std::string, int>> TextFrontend::spell_out(
    const std::string& word
) {
    // Spell out unknown words letter by letter
    // Use letter-to-phoneme rules for basic English
    std::vector<std::pair<std::string, int>> result;
    for (char c : word) {
        // Simple letter → phoneme mapping
        switch (std::tolower(c)) {
            case 'a': result.push_back({"ey", 1}); break;
            case 'b': result.push_back({"b", 0}); break;
            case 'c': result.push_back({"s", 0}); break;
            case 'd': result.push_back({"d", 0}); break;
            case 'e': result.push_back({"iy", 1}); break;
            case 'f': result.push_back({"f", 0}); break;
            case 'g': result.push_back({"g", 0}); break;
            case 'h': result.push_back({"hh", 0}); break;
            case 'i': result.push_back({"ay", 1}); break;
            case 'j': result.push_back({"jh", 0}); break;
            case 'k': result.push_back({"k", 0}); break;
            case 'l': result.push_back({"l", 0}); break;
            case 'm': result.push_back({"m", 0}); break;
            case 'n': result.push_back({"n", 0}); break;
            case 'o': result.push_back({"ow", 1}); break;
            case 'p': result.push_back({"p", 0}); break;
            case 'q': result.push_back({"k", 0}); break;
            case 'r': result.push_back({"r", 0}); break;
            case 's': result.push_back({"s", 0}); break;
            case 't': result.push_back({"t", 0}); break;
            case 'u': result.push_back({"uw", 1}); break;
            case 'v': result.push_back({"V", 0}); break;
            case 'w': result.push_back({"w", 0}); break;
            case 'x': result.push_back({"k", 0}); result.push_back({"s", 0}); break;
            case 'y': result.push_back({"y", 0}); break;
            case 'z': result.push_back({"z", 0}); break;
            default: break;
        }
    }
    return result;
}

int TextFrontend::phoneme_to_id(const std::string& phoneme) const {
    static const std::string phone_prefix = "phone_";
    if (phoneme.rfind(phone_prefix, 0) == 0) {
        return std::atoi(phoneme.c_str() + phone_prefix.size());
    }

    auto it = phoneme_vocab_.find(phoneme);
    if (it != phoneme_vocab_.end()) {
        return it->second;
    }
    // Return UNK id
    auto unk_it = phoneme_vocab_.find("UNK");
    return (unk_it != phoneme_vocab_.end()) ? unk_it->second : 0;
}

void TextFrontend::insert_blanks(std::vector<int32_t>& ids, int32_t blank_id) {
    // insert_blanks([a, b, c], 0) → [0, a, 0, b, 0, c, 0]
    std::vector<int32_t> result;
    result.reserve(ids.size() * 2 + 1);
    result.push_back(blank_id);
    for (size_t i = 0; i < ids.size(); i++) {
        result.push_back(ids[i]);
        result.push_back(blank_id);
    }
    ids = std::move(result);
}

PhonemeResult TextFrontend::process(const std::string& text) {
    PhonemeResult result;

    // 1. Clean text
    std::string cleaned = clean_text(text);

    // 2. Normalize (lowercase, expand abbreviations, normalize numbers)
    std::string normalized = normalize_text(cleaned);

    // 3. Tokenize and look up phonemes. Keep punctuation as model tokens
    // instead of attaching it to neighboring words.
    std::vector<std::string> phones;
    std::vector<int32_t> tones;

    // Add start token
    phones.push_back("_");
    tones.push_back(0);

    auto flush_word = [&](std::string& word) {
        if (word.empty()) {
            return;
        }
        auto word_phones = word_to_phonemes(word);
        for (const auto& [ph, tone] : word_phones) {
            phones.push_back(ph);
            tones.push_back(tone);
        }
        word.clear();
    };

    auto is_punctuation = [](unsigned char c) {
        return c == '!' || c == '?' || c == ',' || c == '.' || c == '\'' || c == '-';
    };

    std::string word;
    for (unsigned char c : normalized) {
        if (std::isalnum(c)) {
            word.push_back((char)c);
        } else if (std::isspace(c)) {
            flush_word(word);
        } else if (is_punctuation(c)) {
            flush_word(word);
            phones.push_back(std::string(1, (char)c));
            tones.push_back(0);
        }
    }
    flush_word(word);

    // Add end token
    phones.push_back("_");
    tones.push_back(0);

    // 4. Convert to IDs
    for (const auto& ph : phones) {
        result.phone_ids.push_back(phoneme_to_id(ph));
    }

    // Adjust tones for English
    for (int t : tones) {
        result.tone_ids.push_back(t + TONE_OFFSET_EN);
    }

    // Language IDs (all English)
    result.lang_ids.resize(result.phone_ids.size(), LANG_EN);

    // 5. Insert blanks
    insert_blanks(result.phone_ids, 0);
    insert_blanks(result.tone_ids, 0);
    insert_blanks(result.lang_ids, 0);

    return result;
}

} // namespace inflect
