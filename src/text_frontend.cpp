#include "text_frontend.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

namespace inflect {

static bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

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

static bool starts_with_at(const std::string& s, size_t pos, const char* needle) {
    const size_t n = std::strlen(needle);
    return pos + n <= s.size() && std::memcmp(s.data() + pos, needle, n) == 0;
}

static std::string cmudict_index_path(const std::string& path) {
    const std::string suffix = ".bin";
    if (path.size() >= suffix.size() &&
        path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return path.substr(0, path.size() - suffix.size()) + ".idx";
    }
    return path + ".idx";
}

template <typename T>
static bool read_exact(std::ifstream& f, T& value) {
    f.read(reinterpret_cast<char*>(&value), sizeof(value));
    return f.gcount() == static_cast<std::streamsize>(sizeof(value));
}

static bool cleanup_punctuation_at(const std::string& s, size_t pos, size_t& len) {
    if (starts_with_at(s, pos, "\u2026")) {
        len = std::strlen("\u2026");
        return true;
    }
    const unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c == ',' || c == '.' || c == '!' || c == '?') {
        len = 1;
        return true;
    }
    return false;
}

static bool has_word_boundary_before(const std::string& s, size_t pos) {
    if (pos == 0) {
        return true;
    }
    const unsigned char prev = static_cast<unsigned char>(s[pos - 1]);
    return !std::isalnum(prev) && prev != '_';
}

static bool ascii_iequals_at(const std::string& s, size_t pos, const char* needle) {
    const size_t n = std::strlen(needle);
    if (pos + n > s.size()) {
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        const unsigned char a = static_cast<unsigned char>(s[pos + i]);
        const unsigned char b = static_cast<unsigned char>(needle[i]);
        if (std::tolower(a) != std::tolower(b)) {
            return false;
        }
    }
    return true;
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

    if (flash_dict_.is_open()) {
        flash_dict_.close();
    }
    dict_path_ = path;
    dict_count_ = count;
    legacy_phone_ids_ = legacy_phone_ids;
    flash_cmu_ = env_flag_enabled("INFLECT_FLASH_CMU");
#if defined(INFLECT_LOW_MEMORY)
    flash_cmu_ = true;
#endif
    if (flash_cmu_) {
        if (!load_cmudict_sparse_index(f, count)) {
            return false;
        }
        if (legacy_phone_ids_) {
            fprintf(stderr, "[TextFrontend] Remapped legacy cmudict phone IDs to model vocabulary\n");
        }
        fprintf(stderr, "[TextFrontend] Loaded sparse index for %u dictionary entries\n", dict_count_);
        return true;
    }

    dict_.clear();
    dict_words_.clear();
    dict_phones_.clear();
    dict_.reserve(count);
    dict_words_.reserve(count * 8);
    dict_phones_.reserve(count * 6);

    for (uint32_t entry_idx = 0; entry_idx < count && f.good(); entry_idx++) {
        uint8_t word_len = 0;
        f.read(reinterpret_cast<char*>(&word_len), 1);
        if (f.gcount() < 1) break;

        std::string entry_word_text(word_len, '\0');
        f.read(&entry_word_text[0], word_len);

        uint8_t n_phones;
        f.read(reinterpret_cast<char*>(&n_phones), 1);

        DictEntry entry;
        entry.word_offset = static_cast<uint32_t>(dict_words_.size());
        entry.phone_offset = static_cast<uint32_t>(dict_phones_.size());
        entry.word_len = word_len;
        entry.phone_count = n_phones;
        dict_words_.append(entry_word_text);
        for (int i = 0; i < n_phones; i++) {
            uint8_t phone_id = 0;
            uint8_t tone = 0;
            f.read(reinterpret_cast<char*>(&phone_id), 1);
            f.read(reinterpret_cast<char*>(&tone), 1);
            if (legacy_phone_ids) {
                phone_id = remap_legacy_phone_id(phone_id);
            }
            dict_phones_.push_back({phone_id, tone});
        }

        dict_.push_back(entry);
    }

    if (legacy_phone_ids) {
        fprintf(stderr, "[TextFrontend] Remapped legacy cmudict phone IDs to model vocabulary\n");
    }
    fprintf(stderr, "[TextFrontend] Loaded %zu dictionary entries\n", dict_.size());
    return true;
}

bool TextFrontend::load_cmudict_sparse_index(std::ifstream& f, uint32_t count) {
    if (load_cmudict_sidecar_index(count)) {
        return true;
    }

    constexpr uint32_t kIndexStride = 256;
    sparse_stride_ = kIndexStride;
    dict_.clear();
    dict_words_.clear();
    dict_phones_.clear();
    sparse_index_.clear();
    sparse_words_.clear();
    sparse_index_.reserve(count / kIndexStride + 1);
    sparse_words_.reserve((count / kIndexStride + 1) * 8);

    for (uint32_t entry_idx = 0; entry_idx < count && f.good(); entry_idx++) {
        const uint64_t entry_offset = static_cast<uint64_t>(f.tellg());
        uint8_t word_len = 0;
        f.read(reinterpret_cast<char*>(&word_len), 1);
        if (f.gcount() < 1) break;

        char entry_word_text[256];
        f.read(entry_word_text, word_len);

        uint8_t n_phones = 0;
        f.read(reinterpret_cast<char*>(&n_phones), 1);
        f.seekg(static_cast<std::streamoff>(2 * n_phones), std::ios::cur);

        if (entry_idx % kIndexStride == 0) {
            SparseIndexEntry entry;
            entry.word_offset = static_cast<uint32_t>(sparse_words_.size());
            entry.word_len = word_len;
            entry.offset = entry_offset;
            entry.entry_index = entry_idx;
            sparse_words_.append(entry_word_text, word_len);
            sparse_index_.push_back(entry);
        }
    }
    flash_dict_.open(dict_path_, std::ios::binary);
    return !sparse_index_.empty();
}

bool TextFrontend::load_cmudict_sidecar_index(uint32_t count) {
    const std::string index_path = cmudict_index_path(dict_path_);
    std::ifstream index(index_path, std::ios::binary);
    if (!index.is_open()) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t stride = 0;
    uint32_t sidecar_count = 0;
    uint32_t index_count = 0;
    constexpr uint32_t kIndexMagic = 0x31594d43; // "CMY1", little-endian
    if (!read_exact(index, magic) || magic != kIndexMagic ||
        !read_exact(index, stride) || stride == 0 ||
        !read_exact(index, sidecar_count) || sidecar_count != count ||
        !read_exact(index, index_count) || index_count == 0) {
        fprintf(stderr, "[TextFrontend] Ignoring invalid cmudict index: %s\n",
                index_path.c_str());
        return false;
    }

    dict_.clear();
    dict_words_.clear();
    dict_phones_.clear();
    sparse_index_.clear();
    sparse_words_.clear();
    sparse_stride_ = stride;
    sparse_index_.reserve(index_count);
    sparse_words_.reserve(index_count * 8);

    for (uint32_t i = 0; i < index_count; i++) {
        SparseIndexEntry entry;
        uint8_t word_len = 0;
        if (!read_exact(index, entry.entry_index) ||
            !read_exact(index, entry.offset) ||
            !read_exact(index, word_len) ||
            word_len == 0) {
            fprintf(stderr, "[TextFrontend] Ignoring truncated cmudict index: %s\n",
                    index_path.c_str());
            sparse_index_.clear();
            sparse_words_.clear();
            return false;
        }
        char word[256];
        index.read(word, word_len);
        if (index.gcount() != static_cast<std::streamsize>(word_len)) {
            fprintf(stderr, "[TextFrontend] Ignoring truncated cmudict index: %s\n",
                    index_path.c_str());
            sparse_index_.clear();
            sparse_words_.clear();
            return false;
        }
        entry.word_len = word_len;
        entry.word_offset = static_cast<uint32_t>(sparse_words_.size());
        sparse_words_.append(word, word_len);
        sparse_index_.push_back(entry);
    }

    flash_dict_.open(dict_path_, std::ios::binary);
    if (!flash_dict_.is_open()) {
        sparse_index_.clear();
        sparse_words_.clear();
        return false;
    }

    fprintf(stderr,
            "[TextFrontend] Loaded cmudict sidecar index entries=%u stride=%u path=%s\n",
            index_count, stride, index_path.c_str());
    return true;
}

std::string_view TextFrontend::dict_word(const DictEntry& entry) const {
    return std::string_view(dict_words_.data() + entry.word_offset, entry.word_len);
}

const TextFrontend::DictEntry* TextFrontend::find_dict_entry(std::string_view word) const {
    auto it = std::lower_bound(
        dict_.begin(), dict_.end(), word,
        [&](const DictEntry& entry, std::string_view needle) {
            return dict_word(entry) < needle;
        }
    );
    if (it != dict_.end() && dict_word(*it) == word) {
        return &*it;
    }
    return nullptr;
}

std::string_view TextFrontend::sparse_word(const SparseIndexEntry& entry) const {
    return std::string_view(sparse_words_.data() + entry.word_offset, entry.word_len);
}

bool TextFrontend::lookup_flash_entry(
    std::string_view word,
    std::vector<PhoneTone>& phonemes
) const {
    if (sparse_index_.empty()) {
        return false;
    }

    auto it = std::upper_bound(
        sparse_index_.begin(), sparse_index_.end(), word,
        [&](std::string_view needle, const SparseIndexEntry& entry) {
            return needle < sparse_word(entry);
        }
    );
    if (it != sparse_index_.begin()) {
        --it;
    }

    if (!flash_dict_.is_open()) {
        return false;
    }
    flash_dict_.clear();
    flash_dict_.seekg(static_cast<std::streamoff>(it->offset), std::ios::beg);

    const uint32_t end_index = std::min<uint32_t>(
        it->entry_index + sparse_stride_,
        dict_count_
    );
    for (uint32_t entry_idx = it->entry_index; entry_idx < end_index && flash_dict_.good(); entry_idx++) {
        uint8_t word_len = 0;
        flash_dict_.read(reinterpret_cast<char*>(&word_len), 1);
        if (flash_dict_.gcount() < 1) break;

        char entry_word[256];
        flash_dict_.read(entry_word, word_len);

        uint8_t n_phones = 0;
        flash_dict_.read(reinterpret_cast<char*>(&n_phones), 1);

        if (word_len == word.size() && std::memcmp(entry_word, word.data(), word_len) == 0) {
            phonemes.clear();
            phonemes.reserve(n_phones);
            for (int i = 0; i < n_phones; i++) {
                uint8_t phone_id = 0;
                uint8_t tone = 0;
                flash_dict_.read(reinterpret_cast<char*>(&phone_id), 1);
                flash_dict_.read(reinterpret_cast<char*>(&tone), 1);
                if (legacy_phone_ids_) {
                    phone_id = remap_legacy_phone_id(phone_id);
                }
                phonemes.push_back({phone_id, tone});
            }
            return true;
        }

        flash_dict_.seekg(static_cast<std::streamoff>(2 * n_phones), std::ios::cur);
    }

    return false;
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

    std::string cleaned;
    cleaned.reserve(result.size());
    bool pending_space = false;
    bool last_was_cleanup_punct = false;
    size_t last_punct_pos = 0;

    for (size_t i = 0; i < result.size();) {
        const unsigned char c = static_cast<unsigned char>(result[i]);
        if (std::isspace(c)) {
            pending_space = true;
            i++;
            continue;
        }

        size_t punct_len = 0;
        if (cleanup_punctuation_at(result, i, punct_len)) {
            if (last_was_cleanup_punct) {
                cleaned.erase(last_punct_pos);
            }
            if (!cleaned.empty() && cleaned.back() == ' ') {
                cleaned.pop_back();
            }
            last_punct_pos = cleaned.size();
            cleaned.append(result, i, punct_len);
            last_was_cleanup_punct = true;
            pending_space = false;
            i += punct_len;
            continue;
        }

        if (pending_space && !cleaned.empty()) {
            cleaned.push_back(' ');
        }
        pending_space = false;
        last_was_cleanup_punct = false;
        cleaned.push_back(result[i]);
        i++;
    }

    return cleaned;
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
    std::string result = text;
    struct Abbrev {
        const char* pattern;
        const char* expansion;
    };
    static constexpr Abbrev abbrevs[] = {
        {"mrs.", "misess"}, {"mr.", "mister"}, {"dr.", "doctor"},
        {"st.", "saint"}, {"co.", "company"}, {"jr.", "junior"},
        {"maj.", "major"}, {"gen.", "general"}, {"drs.", "doctors"},
        {"rev.", "reverend"}, {"lt.", "lieutenant"}, {"hon.", "honorable"},
        {"sgt.", "sergeant"}, {"capt.", "captain"}, {"esq.", "esquire"},
        {"ltd.", "limited"}, {"col.", "colonel"}, {"ft.", "fort"},
    };
    for (const auto& abbrev : abbrevs) {
        const size_t pattern_len = std::strlen(abbrev.pattern);
        for (size_t pos = 0; pos < result.size();) {
            if (has_word_boundary_before(result, pos) &&
                ascii_iequals_at(result, pos, abbrev.pattern)) {
                result.replace(pos, pattern_len, abbrev.expansion);
                pos += std::strlen(abbrev.expansion);
            } else {
                pos++;
            }
        }
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

    std::vector<PhoneTone> flash_phonemes;
    if (flash_cmu_ && lookup_flash_entry(upper_word, flash_phonemes)) {
        for (const auto& phone : flash_phonemes) {
            result.push_back({"phone_" + std::to_string(phone.phone_id), phone.tone});
        }
        return result;
    }

    const DictEntry* entry = find_dict_entry(upper_word);
    if (entry) {
        for (int i = 0; i < entry->phone_count; i++) {
            const auto& [phone_id, tone] = dict_phones_[entry->phone_offset + i];
            // Convert phone_id back to symbol (reverse lookup)
            // For now, return placeholder
            result.push_back({"phone_" + std::to_string(phone_id), tone});
        }
        return result;
    }

    // OOV: spell out
    return spell_out(word);
}

std::vector<TextFrontend::PhoneTone> TextFrontend::word_to_phone_ids(
    const std::string& word
) {
    std::vector<PhoneTone> result;

    if (word.size() == 1) {
        auto it = phoneme_vocab_.find(word);
        if (it != phoneme_vocab_.end()) {
            result.push_back({it->second, 0});
            return result;
        }
    }

    std::string upper_word = word;
    std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (flash_cmu_ && lookup_flash_entry(upper_word, result)) {
        return result;
    }

    const DictEntry* entry = find_dict_entry(upper_word);
    if (entry) {
        result.reserve(entry->phone_count);
        for (int i = 0; i < entry->phone_count; i++) {
            const auto& [phone_id, tone] = dict_phones_[entry->phone_offset + i];
            result.push_back({phone_id, tone});
        }
        return result;
    }

    auto spelled = spell_out(word);
    result.reserve(spelled.size());
    for (const auto& [phone, tone] : spelled) {
        result.push_back({phoneme_to_id(phone), tone});
    }
    return result;
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
    std::vector<int32_t> phone_ids;
    std::vector<int32_t> tones;

    // Add start token
    phone_ids.push_back(0);
    tones.push_back(0);

    auto flush_word = [&](std::string& word) {
        if (word.empty()) {
            return;
        }
        auto word_phones = word_to_phone_ids(word);
        for (const auto& phone : word_phones) {
            phone_ids.push_back(phone.phone_id);
            tones.push_back(phone.tone);
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
            phone_ids.push_back(phoneme_to_id(std::string(1, (char)c)));
            tones.push_back(0);
        }
    }
    flush_word(word);

    // Add end token
    phone_ids.push_back(0);
    tones.push_back(0);

    // 4. Store IDs
    result.phone_ids = std::move(phone_ids);

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
