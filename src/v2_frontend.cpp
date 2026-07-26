#include "v2_frontend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

#if defined(INFLECT_V2_ESPEAK_FALLBACK)
#if defined(ESP_PLATFORM) || defined(ARDUINO)
#error "INFLECT_V2_ESPEAK_FALLBACK is CPU-only and must not be enabled for ESP32"
#endif
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace inflect {
namespace {

constexpr std::array<char, 4> kLexiconMagic = {'I', 'V', 'L', '2'};
constexpr std::array<char, 4> kIndexMagic = {'I', 'V', 'I', '2'};
constexpr uint16_t kLexiconVersion = 1;
constexpr uint16_t kHeaderSize = 76;

template <typename T>
bool read_le(std::istream& in, T& value) {
    std::array<unsigned char, sizeof(T)> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (in.gcount() != static_cast<std::streamsize>(bytes.size())) return false;
    using U = typename std::make_unsigned<T>::type;
    U result = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        result |= static_cast<U>(bytes[i]) << (i * 8);
    }
    value = static_cast<T>(result);
    return true;
}

bool read_bytes(std::istream& in, void* dst, size_t size) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
    return in.gcount() == static_cast<std::streamsize>(size);
}

bool ascii_alpha(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool ascii_digit(char c) {
    return c >= '0' && c <= '9';
}

char ascii_lower(char c) {
    return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c;
}

std::string lower_ascii(std::string_view value) {
    std::string result(value);
    for (char& c : result) c = ascii_lower(c);
    return result;
}

bool boundary_before(std::string_view text, size_t pos) {
    return pos == 0 || !ascii_alpha(text[pos - 1]);
}

bool boundary_after(std::string_view text, size_t pos) {
    return pos >= text.size() || !ascii_alpha(text[pos]);
}

void replace_all(std::string& text, std::string_view from, std::string_view to) {
    for (size_t pos = 0; (pos = text.find(from, pos)) != std::string::npos;) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

void replace_words_ci(std::string& text, std::string_view from, std::string_view to) {
    for (size_t pos = 0; pos + from.size() <= text.size();) {
        bool match = boundary_before(text, pos) &&
                     boundary_after(text, pos + from.size());
        for (size_t i = 0; match && i < from.size(); ++i) {
            match = ascii_lower(text[pos + i]) == ascii_lower(from[i]);
        }
        if (match) {
            text.replace(pos, from.size(), to);
            pos += to.size();
        } else {
            ++pos;
        }
    }
}

std::string under_twenty(unsigned value) {
    static constexpr const char* names[] = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    };
    return names[value];
}

std::string integer_words(uint64_t value) {
    if (value < 20) return under_twenty(static_cast<unsigned>(value));
    if (value < 100) {
        static constexpr const char* tens[] = {
            "", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety",
        };
        const unsigned remainder = static_cast<unsigned>(value % 10);
        return std::string(tens[value / 10]) +
               (remainder ? " " + under_twenty(remainder) : "");
    }
    if (value < 1000) {
        const uint64_t remainder = value % 100;
        return under_twenty(static_cast<unsigned>(value / 100)) + " hundred" +
               (remainder ? " and " + integer_words(remainder) : "");
    }
    static constexpr struct {
        uint64_t scale;
        const char* name;
    } scales[] = {
        {1000000000000ULL, "trillion"},
        {1000000000ULL, "billion"},
        {1000000ULL, "million"},
        {1000ULL, "thousand"},
    };
    for (const auto& scale : scales) {
        if (value >= scale.scale) {
            const uint64_t remainder = value % scale.scale;
            std::string result = integer_words(value / scale.scale) + " " + scale.name;
            if (remainder) {
                result += remainder < 100 ? " and " : " ";
                result += integer_words(remainder);
            }
            return result;
        }
    }
    return {};
}

std::string ordinal_words(unsigned value) {
    static constexpr const char* small[] = {
        "zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
        "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
        "eighteenth", "nineteenth",
    };
    if (value < 20) return small[value];
    if (value < 100 && value % 10) {
        return integer_words(value - value % 10) + " " + small[value % 10];
    }
    static constexpr const char* exact_tens[] = {
        "", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
        "sixtieth", "seventieth", "eightieth", "ninetieth",
    };
    if (value < 100) return exact_tens[value / 10];
    if (value % 100 == 0 && value < 1000) {
        return integer_words(value / 100) + " hundredth";
    }
    return integer_words(value);
}

bool parse_u64(std::string_view value, uint64_t& output) {
    output = 0;
    bool any = false;
    for (char c : value) {
        if (c == ',') continue;
        if (!ascii_digit(c)) return false;
        const unsigned digit = static_cast<unsigned>(c - '0');
        if (output > (std::numeric_limits<uint64_t>::max() - digit) / 10) return false;
        output = output * 10 + digit;
        any = true;
    }
    return any;
}

std::string digit_words(std::string_view digits, bool identifier_zero = false) {
    std::string result;
    size_t digit_index = 0;
    for (char c : digits) {
        if (!ascii_digit(c)) continue;
        if (!result.empty()) result += " ";
        result += (identifier_zero && c == '0' && digit_index > 0)
                      ? "oh"
                      : under_twenty(static_cast<unsigned>(c - '0'));
        ++digit_index;
    }
    return result;
}

bool valid_date(unsigned month, unsigned day, unsigned year) {
    if (month < 1 || month > 12 || day < 1) return false;
    static constexpr unsigned days[] =
        {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    unsigned max_day = days[month - 1];
    const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    if (month == 2 && leap) ++max_day;
    return day <= max_day;
}

std::string cleanup_punctuation(std::string text) {
    std::string output;
    output.reserve(text.size() + 8);
    bool pending_space = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (std::isspace(c)) {
            pending_space = !output.empty();
            continue;
        }
        const bool punctuation = std::strchr(",;:.!?", text[i]) != nullptr;
        if (punctuation && !output.empty() && output.back() == ' ') output.pop_back();
        if (pending_space && !output.empty() && !punctuation) output.push_back(' ');
        pending_space = false;
        output.push_back(text[i]);
        if (punctuation && i + 1 < text.size() &&
            !std::isspace(static_cast<unsigned char>(text[i + 1])) &&
            !std::strchr(",;:.!?", text[i + 1])) {
            output.push_back(' ');
        }
    }
    return output;
}

std::string collapse_spaces(std::string text) {
    std::string output;
    output.reserve(text.size());
    bool pending = false;
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            pending = !output.empty();
        } else {
            if (pending) output.push_back(' ');
            output.push_back(static_cast<char>(c));
            pending = false;
        }
    }
    return output;
}

size_t utf8_codepoint_size(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xe0) == 0xc0) return 2;
    if ((lead & 0xf0) == 0xe0) return 3;
    if ((lead & 0xf8) == 0xf0) return 4;
    return 1;
}

#if defined(INFLECT_V2_ESPEAK_FALLBACK)
uint32_t utf8_codepoint_value(std::string_view value) {
    if (value.empty()) return 0;
    const auto lead = static_cast<unsigned char>(value[0]);
    if (lead < 0x80) return lead;
    uint32_t result = 0;
    size_t continuation = 0;
    if ((lead & 0xe0) == 0xc0) {
        result = lead & 0x1f;
        continuation = 1;
    } else if ((lead & 0xf0) == 0xe0) {
        result = lead & 0x0f;
        continuation = 2;
    } else if ((lead & 0xf8) == 0xf0) {
        result = lead & 0x07;
        continuation = 3;
    } else {
        return 0;
    }
    if (value.size() < continuation + 1) return 0;
    for (size_t i = 1; i <= continuation; ++i) {
        const auto byte = static_cast<unsigned char>(value[i]);
        if ((byte & 0xc0) != 0x80) return 0;
        result = (result << 6) | (byte & 0x3f);
    }
    return result;
}

bool espeak_oov_tokens(std::string_view word, std::vector<uint8_t>& tokens) {
    tokens.clear();
    if (word.empty()) return false;
    int output_pipe[2] = {-1, -1};
    if (pipe(output_pipe) != 0) return false;

    const char* executable = std::getenv("INFLECT_V2_ESPEAK_BIN");
    if (!executable || !executable[0]) executable = "espeak-ng";
    std::string input(word);
    char* arguments[] = {
        const_cast<char*>(executable),
        const_cast<char*>("-q"),
        const_cast<char*>("--ipa=3"),
        const_cast<char*>("--sep="),
        input.data(),
        nullptr,
    };
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
    posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, output_pipe[1]);
    pid_t child = 0;
    const int spawn_result = posix_spawnp(
        &child, executable, &actions, nullptr, arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(output_pipe[1]);
    if (spawn_result != 0) {
        close(output_pipe[0]);
        return false;
    }

    std::string phonemes;
    std::array<char, 256> buffer{};
    while (true) {
        const ssize_t count = read(output_pipe[0], buffer.data(), buffer.size());
        if (count <= 0) break;
        phonemes.append(buffer.data(), static_cast<size_t>(count));
    }
    close(output_pipe[0]);
    int status = 0;
    if (waitpid(child, &status, 0) < 0 ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return false;
    }

    for (size_t i = 0; i < phonemes.size();) {
        const unsigned char lead = static_cast<unsigned char>(phonemes[i]);
        if (std::isspace(lead) || phonemes[i] == '_') {
            ++i;
            continue;
        }
        const size_t size =
            std::min(utf8_codepoint_size(lead), phonemes.size() - i);
        const std::string_view codepoint(phonemes.data() + i, size);
        const int id = v2::symbol_id(codepoint);
        if (id > 0) {
            tokens.push_back(static_cast<uint8_t>(id));
        } else {
            const uint32_t value = utf8_codepoint_value(codepoint);
            // eSpeak sometimes emits combining pronunciation detail absent
            // from the released 178-position model table.
            if (value < 0x0300 || value > 0x036f) {
                tokens.clear();
                return false;
            }
        }
        i += size;
    }
    return !tokens.empty();
}
#endif

bool append_with_space(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src) {
    if (src.empty()) return false;
    if (!dst.empty() && dst.back() != v2::kSpaceId) dst.push_back(v2::kSpaceId);
    dst.insert(dst.end(), src.begin(), src.end());
    return true;
}

} // namespace

V2Frontend::~V2Frontend() = default;

bool V2Frontend::load_lexicon(const std::string& bin_path,
                              const std::string& index_path) {
    lexicon_.close();
    sparse_.clear();
    entry_count_ = 0;
    lexicon_path_ = bin_path;
    lexicon_.open(bin_path, std::ios::binary);
    if (!lexicon_.is_open()) {
        std::fprintf(stderr, "[V2Frontend] Failed to open lexicon: %s\n", bin_path.c_str());
        return false;
    }

    std::array<char, 4> magic{};
    uint16_t version = 0;
    uint16_t header_size = 0;
    std::array<uint8_t, 32> symbol_hash{};
    uint32_t sparse_count = 0;
    if (!read_bytes(lexicon_, magic.data(), magic.size()) ||
        magic != kLexiconMagic ||
        !read_le(lexicon_, version) || version != kLexiconVersion ||
        !read_le(lexicon_, header_size) || header_size < kHeaderSize ||
        !read_bytes(lexicon_, symbol_hash.data(), symbol_hash.size()) ||
        symbol_hash != v2::kSymbolHash ||
        !read_le(lexicon_, entry_count_) || entry_count_ == 0 ||
        !read_le(lexicon_, sparse_stride_) || sparse_stride_ == 0 ||
        !read_le(lexicon_, sparse_count) ||
        !read_le(lexicon_, entries_offset_) ||
        !read_le(lexicon_, index_offset_)) {
        std::fprintf(stderr, "[V2Frontend] Invalid or incompatible IVL2 header: %s\n",
                     bin_path.c_str());
        lexicon_.close();
        entry_count_ = 0;
        return false;
    }
    std::array<char, 8> reserved{};
    if (!read_bytes(lexicon_, reserved.data(), reserved.size())) return false;

    bool loaded_index = index_offset_ != 0 && load_embedded_sparse_index();
    const std::string sidecar = index_path.empty() ? bin_path + ".idx" : index_path;
    bool loaded_sidecar = false;
    std::string alternate_sidecar;
    if (!loaded_index) {
        loaded_sidecar = load_sidecar(sidecar);
    }
    if (!loaded_index && !loaded_sidecar && index_path.empty()) {
        alternate_sidecar = bin_path;
        const size_t extension = alternate_sidecar.rfind('.');
        if (extension != std::string::npos) {
            alternate_sidecar.resize(extension);
        }
        alternate_sidecar += ".idx";
        if (alternate_sidecar != sidecar) {
            loaded_sidecar = load_sidecar(alternate_sidecar);
        }
    }
    if (!loaded_index && !loaded_sidecar) {
        std::fprintf(stderr,
                     "[V2Frontend] Missing or invalid embedded/IVI2 index: %s%s%s\n",
                     sidecar.c_str(),
                     alternate_sidecar.empty() ? "" : " or ",
                     alternate_sidecar.c_str());
        lexicon_.close();
        entry_count_ = 0;
        return false;
    }
    if (sparse_.size() != sparse_count) {
        std::fprintf(stderr,
                     "[V2Frontend] Sparse count mismatch header=%u index=%zu\n",
                     sparse_count, sparse_.size());
        return false;
    }
    std::fprintf(stderr,
                 "[V2Frontend] Loaded IVL2 entries=%u sparse=%zu stride=%u symbols=%s\n",
                 entry_count_, sparse_.size(), sparse_stride_, v2::kSymbolHashHex);
    return true;
}

bool V2Frontend::load_sidecar(const std::string& path) {
    std::ifstream index(path, std::ios::binary);
    if (!index.is_open()) return false;
    std::array<char, 4> magic{};
    uint16_t version = 0;
    uint16_t header_size = 0;
    std::array<uint8_t, 32> hash{};
    uint32_t entry_count = 0;
    uint32_t stride = 0;
    uint32_t count = 0;
    if (!read_bytes(index, magic.data(), magic.size()) || magic != kIndexMagic ||
        !read_le(index, version) || version != kLexiconVersion ||
        !read_le(index, header_size) || header_size < 52 ||
        !read_bytes(index, hash.data(), hash.size()) || hash != v2::kSymbolHash ||
        !read_le(index, entry_count) || entry_count != entry_count_ ||
        !read_le(index, stride) || stride != sparse_stride_ ||
        !read_le(index, count) || count == 0) {
        return false;
    }
    sparse_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        SparseEntry item;
        uint16_t word_size = 0;
        if (!read_le(index, item.entry_index) ||
            !read_le(index, item.offset) ||
            !read_le(index, word_size) || word_size == 0) {
            sparse_.clear();
            return false;
        }
        item.word.resize(word_size);
        if (!read_bytes(index, item.word.data(), word_size)) {
            sparse_.clear();
            return false;
        }
        sparse_.push_back(std::move(item));
    }
    return true;
}

bool V2Frontend::load_embedded_sparse_index() {
    if (index_offset_ == 0) return false;
    lexicon_.clear();
    lexicon_.seekg(static_cast<std::streamoff>(index_offset_), std::ios::beg);
    if (!lexicon_) return false;
    std::array<char, 4> magic{};
    uint16_t version = 0;
    uint16_t header_size = 0;
    std::array<uint8_t, 32> hash{};
    uint32_t entry_count = 0;
    uint32_t stride = 0;
    uint32_t count = 0;
    if (!read_bytes(lexicon_, magic.data(), magic.size()) ||
        magic != kIndexMagic ||
        !read_le(lexicon_, version) || version != kLexiconVersion ||
        !read_le(lexicon_, header_size) || header_size < 52 ||
        !read_bytes(lexicon_, hash.data(), hash.size()) ||
        hash != v2::kSymbolHash ||
        !read_le(lexicon_, entry_count) || entry_count != entry_count_ ||
        !read_le(lexicon_, stride) || stride != sparse_stride_ ||
        !read_le(lexicon_, count) || count == 0) {
        return false;
    }
    sparse_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        SparseEntry item;
        uint16_t word_size = 0;
        if (!read_le(lexicon_, item.entry_index) ||
            !read_le(lexicon_, item.offset) ||
            !read_le(lexicon_, word_size) || word_size == 0) {
            sparse_.clear();
            return false;
        }
        item.word.resize(word_size);
        if (!read_bytes(lexicon_, item.word.data(), word_size)) {
            sparse_.clear();
            return false;
        }
        sparse_.push_back(std::move(item));
    }
    return true;
}

bool V2Frontend::lookup(std::string_view word,
                        std::vector<uint8_t>& tokens) const {
    tokens.clear();
    if (!loaded() || word.empty() || sparse_.empty()) return false;
    const std::string needle = lower_ascii(word);
    auto it = std::upper_bound(
        sparse_.begin(), sparse_.end(), needle,
        [](const std::string& value, const SparseEntry& entry) {
            return value < entry.word;
        });
    if (it != sparse_.begin()) --it;

    lexicon_.clear();
    lexicon_.seekg(static_cast<std::streamoff>(it->offset), std::ios::beg);
    std::string current;
    const uint32_t end = std::min(entry_count_, it->entry_index + sparse_stride_);
    for (uint32_t index = it->entry_index; index < end; ++index) {
        uint8_t prefix = 0;
        uint8_t suffix_size = 0;
        uint8_t token_count = 0;
        if (!read_le(lexicon_, prefix) || !read_le(lexicon_, suffix_size) ||
            prefix > current.size()) {
            return false;
        }
        current.resize(prefix);
        const size_t old_size = current.size();
        current.resize(old_size + suffix_size);
        if (!read_bytes(lexicon_, current.data() + old_size, suffix_size) ||
            !read_le(lexicon_, token_count)) {
            return false;
        }
        if (current == needle) {
            tokens.resize(token_count);
            if (!read_bytes(lexicon_, tokens.data(), tokens.size())) {
                tokens.clear();
                return false;
            }
            return true;
        }
        lexicon_.seekg(token_count, std::ios::cur);
        if (current > needle) return false;
    }
    return false;
}

bool V2Frontend::lookup_morphology(std::string_view word,
                                   std::vector<uint8_t>& tokens) const {
    struct Ending {
        const char* text;
        enum Kind { S, ED, ING } kind;
    };
    static constexpr Ending endings[] = {
        {"'s", Ending::S}, {"s'", Ending::S}, {"es", Ending::S},
        {"s", Ending::S}, {"ied", Ending::ED}, {"ed", Ending::ED},
        {"ing", Ending::ING},
    };
    for (const auto& ending : endings) {
        const size_t n = std::strlen(ending.text);
        if (word.size() <= n + 1 || word.substr(word.size() - n) != ending.text) continue;
        std::string stem(word.substr(0, word.size() - n));
        if (std::strcmp(ending.text, "ied") == 0) stem += "y";
        if (ending.kind == Ending::ING && stem.size() >= 2 &&
            stem.back() == stem[stem.size() - 2]) {
            stem.pop_back();
        }
        if (!lookup(stem, tokens)) {
            if (ending.kind == Ending::ING && !stem.empty()) {
                std::string with_e = stem + "e";
                if (!lookup(with_e, tokens)) continue;
            } else {
                continue;
            }
        }
        const uint8_t last = tokens.empty() ? 0 : tokens.back();
        if (ending.kind == Ending::S) {
            // sibilants use /ɪz/, unvoiced obstruents /s/, otherwise /z/.
            if (last == 61 || last == 68 || last == 131 || last == 133) {
                tokens.push_back(102); // ɪ
                tokens.push_back(68);  // z
            } else if (last == 48 || last == 53 || last == 58 || last == 62 ||
                       last == 119) {
                tokens.push_back(61);  // s
            } else {
                tokens.push_back(68);  // z
            }
        } else if (ending.kind == Ending::ED) {
            if (last == 46 || last == 62) {
                tokens.push_back(102); // ɪ
                tokens.push_back(46);  // d
            } else if (last == 48 || last == 53 || last == 58 || last == 61 ||
                       last == 119 || last == 131 || last == 133) {
                tokens.push_back(62);  // t
            } else {
                tokens.push_back(46);  // d
            }
        } else {
            tokens.push_back(102); // ɪ
            tokens.push_back(112); // ŋ
        }
        return true;
    }
    return false;
}

bool V2Frontend::spell_word(std::string_view word,
                            std::vector<uint8_t>& tokens) const {
    static constexpr const char* names[] = {
        "ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch", "eye",
        "jay", "kay", "ell", "em", "en", "oh", "pee", "cue", "ar", "ess",
        "tee", "you", "vee", "double", "ex", "why", "zee",
    };
    tokens.clear();
    bool appended = false;
    for (char c : word) {
        c = ascii_lower(c);
        if (c < 'a' || c > 'z') continue;
        std::vector<uint8_t> letter;
        if (!lookup(names[c - 'a'], letter)) continue;
        append_with_space(tokens, letter);
        if (c == 'w') {
            std::vector<uint8_t> you;
            if (lookup("you", you)) append_with_space(tokens, you);
        }
        appended = true;
    }
    return appended;
}

std::string V2Frontend::normalize(const std::string& input) const {
    std::string text = input;
    replace_all(text, "\u2018", "'");
    replace_all(text, "\u2019", "'");
    replace_all(text, "\u201c", "\"");
    replace_all(text, "\u201d", "\"");
    replace_all(text, "\u2013", "-");
    replace_all(text, "\u2014", ", ");
    replace_all(text, "\u2026", "...");
    for (std::string_view mark : {"(", ")", "[", "]", "{", "}"}) {
        replace_all(text, mark, ", ");
    }
    text = collapse_spaces(text);

    static constexpr struct { const char* from; const char* to; } overrides[] = {
        {"Qwen3", "Qwen three"}, {"Qwen", "Qwen"}, {"PyTorch", "pie torch"},
        {"SQLite", "ess cue lite"}, {"USB-C", "you ess bee see"},
        {"RTX 3060", "ar tee ex thirty sixty"},
        {"RTX 3090", "ar tee ex thirty ninety"},
        {"RTX 4090", "ar tee ex forty ninety"},
        {"RTX 5080", "ar tee ex fifty eighty"},
        {"RTX 5090", "ar tee ex fifty ninety"},
        {"Dr.", "doctor"}, {"Mr.", "mister"}, {"Mrs.", "missus"},
        {"Ms.", "miss"}, {"Prof.", "professor"}, {"St.", "saint"},
        {"vs.", "versus"}, {"etc.", "et cetera"},
        {"e.g.", "for example"}, {"i.e.", "that is"},
    };
    for (const auto& item : overrides) replace_words_ci(text, item.from, item.to);

    // IVL2 is built from the legacy compiled CMU word list, which contains no
    // apostrophe-bearing entries. Expand productive contractions before
    // lookup so embedded builds pronounce words instead of spelling letters.
    static constexpr struct { const char* from; const char* to; } contractions[] = {
        {"aren't", "are not"}, {"can't", "can not"},
        {"couldn't", "could not"}, {"didn't", "did not"},
        {"doesn't", "does not"}, {"don't", "do not"},
        {"hadn't", "had not"}, {"hasn't", "has not"},
        {"haven't", "have not"}, {"isn't", "is not"},
        {"mustn't", "must not"}, {"needn't", "need not"},
        {"shan't", "shall not"}, {"shouldn't", "should not"},
        {"wasn't", "was not"}, {"weren't", "were not"},
        {"won't", "will not"}, {"wouldn't", "would not"},
        {"ain't", "is not"},
        {"i'm", "i am"}, {"i'll", "i will"}, {"i'd", "i would"},
        {"i've", "i have"},
        {"you're", "you are"}, {"you'll", "you will"},
        {"you'd", "you would"}, {"you've", "you have"},
        {"he's", "he is"}, {"he'll", "he will"}, {"he'd", "he would"},
        {"she's", "she is"}, {"she'll", "she will"},
        {"she'd", "she would"},
        {"it's", "it is"}, {"it'll", "it will"}, {"it'd", "it would"},
        {"we're", "we are"}, {"we'll", "we will"},
        {"we'd", "we would"}, {"we've", "we have"},
        {"they're", "they are"}, {"they'll", "they will"},
        {"they'd", "they would"}, {"they've", "they have"},
        {"that's", "that is"}, {"there's", "there is"},
        {"here's", "here is"}, {"what's", "what is"},
        {"who's", "who is"}, {"where's", "where is"},
        {"when's", "when is"}, {"why's", "why is"},
        {"how's", "how is"}, {"let's", "let us"},
    };
    for (const auto& item : contractions) {
        replace_words_ci(text, item.from, item.to);
    }

    static constexpr const char* months[] = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    };
    std::string output;
    output.reserve(text.size() * 2);
    for (size_t i = 0; i < text.size();) {
        if (text[i] == '$' && i + 1 < text.size() && ascii_digit(text[i + 1])) {
            size_t end = i + 1;
            while (end < text.size() && (ascii_digit(text[end]) || text[end] == ',')) ++end;
            const size_t whole_end = end;
            size_t frac_start = std::string::npos;
            if (end < text.size() && text[end] == '.' &&
                end + 1 < text.size() && ascii_digit(text[end + 1])) {
                frac_start = ++end;
                while (end < text.size() && ascii_digit(text[end]) && end - frac_start < 2) ++end;
            }
            uint64_t dollars = 0;
            parse_u64(std::string_view(text).substr(i + 1, whole_end - i - 1),
                      dollars);
            output += integer_words(dollars);
            output += dollars == 1 ? " dollar" : " dollars";
            if (frac_start != std::string::npos && frac_start < end) {
                unsigned cents = static_cast<unsigned>(text[frac_start] - '0') * 10;
                if (frac_start + 1 < end) cents += static_cast<unsigned>(text[frac_start + 1] - '0');
                if (cents) {
                    output += " and " + integer_words(cents);
                    output += cents == 1 ? " cent" : " cents";
                }
            }
            i = end;
            continue;
        }
        if (ascii_digit(text[i]) && boundary_before(text, i)) {
            const size_t begin = i;
            size_t first_end = i;
            while (first_end < text.size() && ascii_digit(text[first_end])) ++first_end;

            // MM/DD/YYYY.
            if (first_end < text.size() && text[first_end] == '/') {
                size_t second_begin = first_end + 1;
                size_t second_end = second_begin;
                while (second_end < text.size() && ascii_digit(text[second_end])) ++second_end;
                size_t third_begin = second_end + 1;
                size_t third_end = third_begin;
                if (second_end < text.size() && text[second_end] == '/') {
                    while (third_end < text.size() && ascii_digit(text[third_end])) ++third_end;
                    uint64_t month = 0, day = 0, year = 0;
                    if (parse_u64(std::string_view(text).substr(begin, first_end - begin), month) &&
                        parse_u64(std::string_view(text).substr(second_begin, second_end - second_begin), day) &&
                        parse_u64(std::string_view(text).substr(third_begin, third_end - third_begin), year) &&
                        valid_date(month, day, year)) {
                        output += months[month - 1];
                        output += " " + ordinal_words(static_cast<unsigned>(day));
                        output += " " + integer_words(year);
                        i = third_end;
                        continue;
                    }
                }
            }

            // Clock time.
            if (first_end < text.size() && text[first_end] == ':') {
                size_t minute_begin = first_end + 1;
                size_t minute_end = minute_begin;
                while (minute_end < text.size() && ascii_digit(text[minute_end])) ++minute_end;
                uint64_t hour = 0, minute = 0;
                if (minute_end - minute_begin == 2 &&
                    parse_u64(std::string_view(text).substr(begin, first_end - begin), hour) &&
                    parse_u64(std::string_view(text).substr(minute_begin, 2), minute) &&
                    hour <= 23 && minute <= 59) {
                    output += integer_words(hour);
                    if (minute == 0) output += " o clock";
                    else if (minute < 10) output += " oh " + integer_words(minute);
                    else output += " " + integer_words(minute);
                    i = minute_end;
                    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
                    if (i < text.size() && (text[i] == 'a' || text[i] == 'A' ||
                                            text[i] == 'p' || text[i] == 'P')) {
                        const char ap = ascii_lower(text[i++]);
                        if (i < text.size() && text[i] == '.') ++i;
                        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
                        if (i < text.size() && (text[i] == 'm' || text[i] == 'M')) {
                            ++i;
                            if (i < text.size() && text[i] == '.') ++i;
                            output += ap == 'a' ? " a m" : " p m";
                        }
                    }
                    continue;
                }
            }

            // Phone suffix NNN-NNNN.
            if (first_end - begin == 3 && first_end < text.size() && text[first_end] == '-') {
                size_t right = first_end + 1;
                while (right < text.size() && ascii_digit(text[right])) ++right;
                if (right - first_end - 1 == 4) {
                    output += digit_words(std::string_view(text).substr(begin, 3));
                    output += ", ";
                    output += digit_words(std::string_view(text).substr(first_end + 1, 4));
                    i = right;
                    continue;
                }
            }

            // Decimal or dotted version.
            if (first_end < text.size() && text[first_end] == '.' &&
                first_end + 1 < text.size() && ascii_digit(text[first_end + 1])) {
                std::vector<std::string_view> parts;
                size_t end = begin;
                while (true) {
                    size_t part_end = end;
                    while (part_end < text.size() && ascii_digit(text[part_end])) ++part_end;
                    parts.push_back(std::string_view(text).substr(end, part_end - end));
                    if (part_end >= text.size() || text[part_end] != '.' ||
                        part_end + 1 >= text.size() || !ascii_digit(text[part_end + 1])) {
                        end = part_end;
                        break;
                    }
                    end = part_end + 1;
                }
                if (parts.size() > 2) {
                    for (size_t p = 0; p < parts.size(); ++p) {
                        uint64_t value = 0;
                        parse_u64(parts[p], value);
                        if (p) output += " point ";
                        output += integer_words(value);
                    }
                } else {
                    uint64_t whole = 0;
                    parse_u64(parts[0], whole);
                    output += integer_words(whole) + " point " + digit_words(parts[1]);
                }
                i = end;
                continue;
            }

            // Ordinal.
            size_t end = first_end;
            if (end + 1 < text.size()) {
                const std::string suffix = lower_ascii(std::string_view(text).substr(end, 2));
                if (suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th") {
                    uint64_t value = 0;
                    parse_u64(std::string_view(text).substr(begin, first_end - begin), value);
                    output += ordinal_words(static_cast<unsigned>(value));
                    i = end + 2;
                    continue;
                }
            }

            while (end < text.size() && (ascii_digit(text[end]) || text[end] == ',')) ++end;
            std::string_view digits = std::string_view(text).substr(begin, end - begin);
            uint64_t value = 0;
            if (parse_u64(digits, value)) {
                size_t digit_count = 0;
                for (char c : digits) digit_count += ascii_digit(c);
                output += (digit_count >= 5 && !(digit_count == 4 && digits.substr(0, 2) == "20"))
                              ? digit_words(digits)
                              : integer_words(value);
                i = end;
                continue;
            }
        }

        // Published acronym behavior: two or more consecutive uppercase ASCII
        // letters are expanded to their spoken letter names.
        if (text[i] >= 'A' && text[i] <= 'Z' && boundary_before(text, i)) {
            size_t end = i;
            while (end < text.size() && text[end] >= 'A' && text[end] <= 'Z') ++end;
            if (end - i >= 2 && boundary_after(text, end)) {
                static constexpr const char* names[] = {
                    "ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch",
                    "eye", "jay", "kay", "ell", "em", "en", "oh", "pee",
                    "cue", "ar", "ess", "tee", "you", "vee", "double you",
                    "ex", "why", "zee",
                };
                for (size_t p = i; p < end; ++p) {
                    if (p != i) output += " ";
                    output += names[text[p] - 'A'];
                }
                i = end;
                continue;
            }
        }
        output.push_back(text[i++]);
    }
    return cleanup_punctuation(output);
}

V2FrontendResult V2Frontend::process(const std::string& text) const {
    V2FrontendResult result;
    result.raw_text = text;
    result.normalized_text = normalize(text);

    enum class Previous : uint8_t { None, Word, Punctuation };
    Previous previous = Previous::None;
    for (size_t i = 0; i < result.normalized_text.size();) {
        const unsigned char lead = static_cast<unsigned char>(result.normalized_text[i]);
        if (std::isspace(lead) || result.normalized_text[i] == '-') {
            ++i;
            continue;
        }
        if (ascii_alpha(result.normalized_text[i]) || result.normalized_text[i] == '\'') {
            const size_t begin = i;
            while (i < result.normalized_text.size() &&
                   (ascii_alpha(result.normalized_text[i]) ||
                    result.normalized_text[i] == '\'')) {
                ++i;
            }
            const std::string word =
                lower_ascii(std::string_view(result.normalized_text).substr(begin, i - begin));
            std::vector<uint8_t> pronunciation;
            if (!lookup(word, pronunciation) &&
                !lookup_morphology(word, pronunciation)) {
                result.oov_words.push_back(word);
                ++oov_count_;
#if defined(INFLECT_V2_ESPEAK_FALLBACK)
                if (espeak_oov_tokens(word, pronunciation)) {
                    std::fprintf(
                        stderr, "[V2Frontend] OOV eSpeak fallback: %s\n",
                        word.c_str());
                } else
#endif
                {
                    spell_word(word, pronunciation);
                    std::fprintf(
                        stderr, "[V2Frontend] OOV spelling fallback: %s\n",
                        word.c_str());
                }
            }
            if (!pronunciation.empty()) {
                if (previous != Previous::None &&
                    !result.unblanked_tokens.empty() &&
                    result.unblanked_tokens.back() != v2::kSpaceId) {
                    result.unblanked_tokens.push_back(v2::kSpaceId);
                }
                result.unblanked_tokens.insert(result.unblanked_tokens.end(),
                                               pronunciation.begin(), pronunciation.end());
                previous = Previous::Word;
            }
            continue;
        }

        const size_t size = std::min(utf8_codepoint_size(lead),
                                     result.normalized_text.size() - i);
        const std::string_view cp(result.normalized_text.data() + i, size);
        const int id = v2::symbol_id(cp);
        if (id > 0) {
            if (previous == Previous::Word &&
                !result.unblanked_tokens.empty() &&
                result.unblanked_tokens.back() == v2::kSpaceId) {
                result.unblanked_tokens.pop_back();
            }
            result.unblanked_tokens.push_back(static_cast<uint8_t>(id));
            previous = Previous::Punctuation;
        }
        i += size;
    }
    while (!result.unblanked_tokens.empty() &&
           result.unblanked_tokens.back() == v2::kSpaceId) {
        result.unblanked_tokens.pop_back();
    }
    result.blanked_tokens = v2::intersperse_blanks(result.unblanked_tokens);
    return result;
}

} // namespace inflect
