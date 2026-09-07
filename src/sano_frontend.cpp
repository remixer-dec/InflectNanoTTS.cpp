#include "sano_frontend.h"
#include "sano_config.h"
#include "v2_frontend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cstring>
#include <type_traits>

namespace inflect {
namespace {

constexpr std::array<char, 4> kSnl1Magic{{'S', 'N', 'L', '1'}};
constexpr std::array<char, 4> kSnl2Magic{{'S', 'N', 'L', '2'}};
constexpr uint16_t kVersion = 1;
constexpr uint16_t kHeaderSize = 88;
constexpr uint16_t kBucketSize = 24;
constexpr uint32_t kSnl2MaxBucketSize = 16;
constexpr uint32_t kSnl2BlockSize = 16;
constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

struct Bucket {
    uint64_t hash = 0;
    uint32_t word_offset = 0;
    uint32_t token_offset = 0;
    uint16_t word_size = 0;
    uint16_t token_size = 0;
    uint32_t flags = 0;
};

template <typename T>
bool read_le(std::FILE* file, T& value) {
    std::array<unsigned char, sizeof(T)> bytes{};
    if (std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
        return false;
    }
    using U = typename std::make_unsigned<T>::type;
    U result = 0;
    for (size_t index = 0; index < bytes.size(); ++index) {
        result |= static_cast<U>(bytes[index]) << (index * 8);
    }
    value = static_cast<T>(result);
    return true;
}

bool seek_absolute(std::FILE* file, uint64_t offset) {
    if (!file || offset > static_cast<uint64_t>(LONG_MAX)) return false;
    return std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0;
}

bool read_bucket(std::FILE* file, uint64_t offset, Bucket& bucket) {
    if (!seek_absolute(file, offset)) return false;
    return read_le(file, bucket.hash) &&
           read_le(file, bucket.word_offset) &&
           read_le(file, bucket.token_offset) &&
           read_le(file, bucket.word_size) &&
           read_le(file, bucket.token_size) &&
           read_le(file, bucket.flags);
}

uint32_t decode_u24(const unsigned char* bytes) {
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16);
}

bool read_u24_pair(std::FILE* file, uint64_t offset, uint32_t& first, uint32_t& second) {
    if (!seek_absolute(file, offset)) return false;
    std::array<unsigned char, 6> bytes{};
    if (std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
        return false;
    }
    first = decode_u24(bytes.data());
    second = decode_u24(bytes.data() + 3);
    return true;
}

char ascii_lower(char value) {
    return value >= 'A' && value <= 'Z'
               ? static_cast<char>(value + ('a' - 'A'))
               : value;
}

bool ascii_alpha(char value) {
    value = ascii_lower(value);
    return value >= 'a' && value <= 'z';
}

bool ascii_digit(char value) {
    return value >= '0' && value <= '9';
}

bool ascii_word_byte(unsigned char value) {
    return value >= 0x80 || ascii_alpha(static_cast<char>(value)) ||
           ascii_digit(static_cast<char>(value));
}

bool word_boundary_before(std::string_view text, size_t pos) {
    return pos == 0 || !ascii_word_byte(static_cast<unsigned char>(text[pos - 1]));
}

bool word_boundary_after(std::string_view text, size_t pos) {
    return pos >= text.size() ||
           !ascii_word_byte(static_cast<unsigned char>(text[pos]));
}

void replace_word_ci(std::string& text, std::string_view from,
                     std::string_view to) {
    for (size_t pos = 0; pos + from.size() <= text.size();) {
        bool match = word_boundary_before(text, pos) &&
                     word_boundary_after(text, pos + from.size());
        for (size_t index = 0; match && index < from.size(); ++index) {
            match = ascii_lower(text[pos + index]) == ascii_lower(from[index]);
        }
        if (match) {
            text.replace(pos, from.size(), to);
            pos += to.size();
        } else {
            ++pos;
        }
    }
}

uint64_t hash_word(std::string_view word) {
    uint64_t hash = kFnvOffset;
    for (unsigned char value : word) {
        const unsigned char normalized = value < 0x80
            ? static_cast<unsigned char>(ascii_lower(static_cast<char>(value)))
            : value;
        hash ^= normalized;
        hash *= kFnvPrime;
    }
    return hash;
}

bool equal_stored_word(const char* stored, size_t size, std::string_view query) {
    if (query.size() != size) return false;
    for (size_t index = 0; index < size; ++index) {
        const unsigned char q = static_cast<unsigned char>(query[index]);
        const char normalized = q < 0x80
            ? ascii_lower(static_cast<char>(q))
            : static_cast<char>(q);
        if (stored[index] != normalized) return false;
    }
    return true;
}

bool hex_nibble(char value, uint8_t& out) {
    if (value >= '0' && value <= '9') {
        out = static_cast<uint8_t>(value - '0');
        return true;
    }
    value = ascii_lower(value);
    if (value >= 'a' && value <= 'f') {
        out = static_cast<uint8_t>(10 + value - 'a');
        return true;
    }
    return false;
}

bool decode_hash(std::string_view hex, std::array<uint8_t, 32>& out) {
    if (hex.size() != out.size() * 2) return false;
    for (size_t index = 0; index < out.size(); ++index) {
        uint8_t hi = 0;
        uint8_t lo = 0;
        if (!hex_nibble(hex[index * 2], hi) ||
            !hex_nibble(hex[index * 2 + 1], lo)) {
            return false;
        }
        out[index] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

void replace_all(std::string& text, const char* from, const char* to) {
    const size_t from_size = std::strlen(from);
    const size_t to_size = std::strlen(to);
    if (from_size == 0) return;
    for (size_t pos = 0; (pos = text.find(from, pos)) != std::string::npos;) {
        text.replace(pos, from_size, to);
        pos += to_size;
    }
}

void append_ids(std::vector<uint8_t>& output, const std::vector<uint8_t>& ids) {
    output.insert(output.end(), ids.begin(), ids.end());
}

} // namespace

SanoFrontend::~SanoFrontend() {
    if (file_) std::fclose(file_);
}

bool SanoFrontend::load_lexicon(
    const std::string& path,
    const SanoPiperConfig& config
) {
    if (file_) {
        std::fclose(file_);
        file_ = nullptr;
    }
    entry_count_ = 0;
    bucket_count_ = 0;
    block_count_ = 0;
    snl2_ = false;
    word_scratch_.clear();
    block_scratch_.clear();
    path_ = path;

    file_ = std::fopen(path.c_str(), "rb");
    if (!file_) {
        std::fprintf(stderr, "[SanoFrontend] failed to open lexicon: %s\n",
                     path.c_str());
        return false;
    }

    std::array<char, 4> magic{};
    uint16_t version = 0;
    uint16_t header_size = 0;
    std::array<uint8_t, 32> stored_hash{};
    std::array<char, 8> language{};
    uint32_t flags = 0;
    if (std::fread(magic.data(), 1, magic.size(), file_) != magic.size() ||
        !read_le(file_, version) || !read_le(file_, header_size)) {
        std::fprintf(stderr, "[SanoFrontend] truncated SNL1 header: %s\n",
                     path.c_str());
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }

    if (magic == kSnl1Magic) {
        if (std::fread(stored_hash.data(), 1, stored_hash.size(), file_) !=
                stored_hash.size() ||
            !read_le(file_, entry_count_) || !read_le(file_, bucket_count_) ||
            !read_le(file_, bucket_offset_) || !read_le(file_, blob_offset_) ||
            !read_le(file_, file_size_) ||
            std::fread(language.data(), 1, language.size(), file_) != language.size() ||
            !read_le(file_, max_word_bytes_) || !read_le(file_, max_token_bytes_) ||
            !read_le(file_, flags)) {
            std::fprintf(stderr, "[SanoFrontend] truncated SNL1 header: %s\n",
                         path.c_str());
            std::fclose(file_);
            file_ = nullptr;
            return false;
        }
    } else if (magic == kSnl2Magic) {
        snl2_ = true;
        uint32_t bucket_offset = 0;
        uint32_t hash_index_offset = 0;
        uint32_t block_offset = 0;
        uint32_t blob_offset = 0;
        uint32_t file_size = 0;
        if (!read_le(file_, flags) ||
            std::fread(stored_hash.data(), 1, stored_hash.size(), file_) !=
                stored_hash.size() ||
            !read_le(file_, entry_count_) ||
            !read_le(file_, bucket_count_) || !read_le(file_, block_count_) ||
            !read_le(file_, bucket_offset) ||
            !read_le(file_, hash_index_offset) ||
            !read_le(file_, block_offset) || !read_le(file_, blob_offset) ||
            !read_le(file_, file_size) ||
            std::fread(language.data(), 1, language.size(), file_) != language.size()) {
            std::fprintf(stderr, "[SanoFrontend] truncated SNL2 header: %s\n",
                         path.c_str());
            std::fclose(file_);
            file_ = nullptr;
            return false;
        }
        bucket_offset_ = bucket_offset;
        hash_index_offset_ = hash_index_offset;
        block_offset_ = block_offset;
        blob_offset_ = blob_offset;
        file_size_ = file_size;
        uint8_t max_bucket_size = 0;
        uint8_t block_size = 0;
        uint16_t reserved = 0;
        if (!read_le(file_, max_bucket_size) || !read_le(file_, block_size) ||
            !read_le(file_, reserved) || max_bucket_size != kSnl2MaxBucketSize ||
            block_size != kSnl2BlockSize || reserved != 0) {
            std::fprintf(stderr, "[SanoFrontend] invalid SNL2 limits: %s\n",
                         path.c_str());
            std::fclose(file_);
            file_ = nullptr;
            return false;
        }
        max_word_bytes_ = 255;
        max_token_bytes_ = 255;
    } else {
        std::fprintf(stderr, "[SanoFrontend] unsupported lexicon magic: %s\n",
                     path.c_str());
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }

    const long header_end = std::ftell(file_);
    if (std::fseek(file_, 0, SEEK_END) != 0) {
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }
    const long actual_size = std::ftell(file_);
    if (header_end < 0 || actual_size < 0 ||
        static_cast<uint64_t>(actual_size) != file_size_) {
        std::fprintf(stderr, "[SanoFrontend] SNL1 file-size mismatch: %s\n",
                     path.c_str());
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }

    const bool power_of_two =
        bucket_count_ != 0 && (bucket_count_ & (bucket_count_ - 1)) == 0;
    const uint64_t buckets_end =
        bucket_offset_ + static_cast<uint64_t>(bucket_count_) * kBucketSize;
    const bool snl1_valid =
        !snl2_ && version == kVersion && header_size == kHeaderSize &&
        entry_count_ != 0 && power_of_two && bucket_offset_ >= header_size &&
        blob_offset_ >= buckets_end && file_size_ >= blob_offset_ &&
        max_word_bytes_ != 0 && max_token_bytes_ != 0;
    const uint64_t snl2_bucket_end =
        bucket_offset_ + (static_cast<uint64_t>(bucket_count_) + 1) * 3;
    const uint64_t snl2_hash_end =
        hash_index_offset_ + static_cast<uint64_t>(entry_count_) * 5;
    const uint64_t snl2_block_end =
        block_offset_ + (static_cast<uint64_t>(block_count_) + 1) * 3;
    const bool snl2_valid =
        snl2_ && version == kVersion && header_size == kHeaderSize && flags == 0 &&
        entry_count_ != 0 && bucket_count_ != 0 && block_count_ != 0 &&
        bucket_offset_ >= header_size && snl2_bucket_end <= hash_index_offset_ &&
        hash_index_offset_ >= snl2_bucket_end && hash_index_offset_ <= block_offset_ &&
        snl2_hash_end <= block_offset_ && block_offset_ <= blob_offset_ &&
        snl2_block_end <= blob_offset_ &&
        blob_offset_ <= file_size_;
    if ((!snl1_valid && !snl2_valid) ||
        (snl2_ && file_size_ > 0xFFFFFFFFULL)) {
        std::fprintf(stderr,
                     "[SanoFrontend] invalid/incompatible lexicon header: %s\n",
                     path.c_str());
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }

    const size_t language_size =
        static_cast<size_t>(std::find(language.begin(), language.end(), '\0') -
                            language.begin());
    language_.assign(language.data(), language_size);
    if (!language_.empty() && language_ != config.language) {
        std::fprintf(stderr,
                     "[SanoFrontend] lexicon language=%s model language=%s\n",
                     language_.c_str(), config.language.c_str());
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }
    // The map hash is informational.  Do not hash or compare it at runtime.
    map_hash_ = config.frontend_hash;
    // SNL2 files are validated offline. Runtime only checks the header,
    // language, exact file size, and that all indexed sections fit in it.
    word_scratch_.reserve(max_word_bytes_);
    block_scratch_.reserve(snl2_ ? (3 + 255 + 255) * 16 : 0);

    std::fprintf(stderr,
                 "[SanoFrontend] loaded %s entries=%u buckets=%u language=%s\n",
                 snl2_ ? "SNL2" : "SNL1",
                 static_cast<unsigned>(entry_count_),
                 static_cast<unsigned>(bucket_count_),
                 language_.c_str());
    return true;
}

bool SanoFrontend::lookup(
    std::string_view word,
    std::vector<uint8_t>& ids
) const {
    ids.clear();
    if (!loaded() || word.empty() || word.size() > max_word_bytes_) return false;

    const uint64_t raw_hash = hash_word(word);
    const uint64_t hash = snl2_ || raw_hash != 0 ? raw_hash : 1;
    if (snl2_) {
        const uint32_t bucket_index =
            static_cast<uint32_t>(hash % bucket_count_);
        uint32_t start = 0;
        uint32_t end = 0;
        if (!read_u24_pair(file_, bucket_offset_ + bucket_index * 3, start, end)) {
            return false;
        }
        if (start > end || end > entry_count_ ||
            end - start > kSnl2MaxBucketSize) {
            return false;
        }
        // An SNL2 bucket has at most 16 records (80 bytes). Reading it once
        // removes the SD seek for every binary-search comparison/collision.
        if (start == end) return false;
        std::array<unsigned char, kSnl2MaxBucketSize * 5> records{};
        const size_t record_bytes = (end - start) * 5;
        if (!seek_absolute(file_, hash_index_offset_ + start * 5) ||
            std::fread(records.data(), 1, record_bytes, file_) != record_bytes) return false;
        const uint16_t fingerprint = static_cast<uint16_t>(hash >> 48);
        for (uint32_t index = 0; index < end - start; ++index) {
            const unsigned char* record = records.data() + index * 5;
            const uint16_t value = static_cast<uint16_t>(record[0]) |
                                   (static_cast<uint16_t>(record[1]) << 8);
            if (value < fingerprint) continue;
            if (value > fingerprint) break;
            const uint32_t locator = decode_u24(record + 2);
            const uint32_t block = locator >> 4;
            const uint32_t ordinal = locator & 0x0F;
            if (block >= block_count_ || ordinal >= 16) return false;
            const uint64_t block_entry_start = static_cast<uint64_t>(block) * 16;
            if (block_entry_start >= entry_count_ ||
                ordinal >= std::min<uint64_t>(16, entry_count_ - block_entry_start)) {
                return false;
            }
            uint32_t block_start = 0;
            uint32_t block_end = 0;
            if (!read_u24_pair(file_, block_offset_ + block * 3, block_start, block_end)) {
                return false;
            }
            if (block_end < block_start || blob_offset_ + block_end > file_size_) {
                return false;
            }
            if (block_end < block_start || block_end - block_start > 0xFFFFFF ||
                !seek_absolute(file_, blob_offset_ + block_start)) return false;
            block_scratch_.resize(block_end - block_start);
            if (std::fread(block_scratch_.data(), 1, block_scratch_.size(), file_) !=
                block_scratch_.size()) return false;
            const uint8_t* cursor = block_scratch_.data();
            const uint8_t* block_limit = cursor + block_scratch_.size();
            word_scratch_.clear();
            for (uint32_t entry = 0; entry <= ordinal; ++entry) {
                if (cursor + 3 > block_limit) return false;
                const uint8_t prefix = cursor[0];
                const uint8_t suffix_size = cursor[1];
                const uint8_t token_size = cursor[2];
                cursor += 3;
                if (prefix > word_scratch_.size() ||
                    cursor + suffix_size + token_size > block_limit) {
                    return false;
                }
                word_scratch_.resize(prefix + suffix_size);
                std::memmove(word_scratch_.data() + prefix, cursor, suffix_size);
                cursor += suffix_size;
                if (entry == ordinal) {
                    if (!equal_stored_word(word_scratch_.data(), word_scratch_.size(),
                                           word)) {
                        continue;
                    }
                    ids.resize(token_size);
                    std::memcpy(ids.data(), cursor, token_size);
                    return !ids.empty();
                }
                cursor += token_size;
            }
        }
        return false;
    }
    uint32_t bucket_index =
        static_cast<uint32_t>(hash) & (bucket_count_ - 1);
    Bucket bucket;
    for (uint32_t probe = 0; probe < bucket_count_; ++probe) {
        const uint64_t position =
            bucket_offset_ + static_cast<uint64_t>(bucket_index) * kBucketSize;
        if (!read_bucket(file_, position, bucket)) return false;
        if (bucket.hash == 0) return false;
        if (bucket.hash == hash && bucket.word_size == word.size() &&
            bucket.word_size <= max_word_bytes_ &&
            bucket.token_size <= max_token_bytes_) {
            word_scratch_.resize(bucket.word_size);
            const uint64_t word_position = blob_offset_ + bucket.word_offset;
            if (word_position + bucket.word_size > file_size_ ||
                !seek_absolute(file_, word_position) ||
                std::fread(word_scratch_.data(), 1, bucket.word_size, file_) !=
                    bucket.word_size) {
                return false;
            }
            if (equal_stored_word(word_scratch_.data(), bucket.word_size, word)) {
                const uint64_t token_position = blob_offset_ + bucket.token_offset;
                if (token_position + bucket.token_size > file_size_) return false;
                ids.resize(bucket.token_size);
                if (!seek_absolute(file_, token_position) ||
                    std::fread(ids.data(), 1, ids.size(), file_) != ids.size()) {
                    ids.clear();
                    return false;
                }
                return !ids.empty();
            }
        }
        bucket_index = (bucket_index + 1) & (bucket_count_ - 1);
    }
    return false;
}

bool SanoFrontend::spell_ascii_word(
    std::string_view word,
    const SanoPiperConfig& config,
    std::vector<uint8_t>& ids
) const {
    static constexpr const char* names[] = {
        "ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch", "eye",
        "jay", "kay", "ell", "em", "en", "oh", "pee", "cue", "ar", "ess",
        "tee", "you", "vee", "double", "ex", "why", "zee",
    };
    ids.clear();
    std::vector<uint8_t> part;
    bool any = false;
    for (char value : word) {
        value = ascii_lower(value);
        if (value < 'a' || value > 'z') return false;
        if (any && config.space_id >= 0 && config.space_id <= 255) {
            ids.push_back(static_cast<uint8_t>(config.space_id));
        }
        if (!lookup(names[value - 'a'], part)) return false;
        append_ids(ids, part);
        if (value == 'w') {
            if (config.space_id >= 0 && config.space_id <= 255) {
                ids.push_back(static_cast<uint8_t>(config.space_id));
            }
            if (!lookup("you", part)) return false;
            append_ids(ids, part);
        }
        any = true;
    }
    return any;
}

std::string SanoFrontend::normalize(const std::string& input) const {
    std::string text = input;
    replace_all(text, "\xE2\x80\x98", "'");
    replace_all(text, "\xE2\x80\x99", "'");
    replace_all(text, "\xE2\x80\x9C", "\"");
    replace_all(text, "\xE2\x80\x9D", "\"");
    replace_all(text, "\xE2\x80\x93", "-");
    replace_all(text, "\xE2\x80\x94", "-");
    replace_all(text, "\xE2\x80\xA6", "...");

    std::string output;
    output.reserve(text.size());
    bool pending_space = false;
    for (unsigned char value : text) {
        if (std::isspace(value)) {
            pending_space = !output.empty();
            continue;
        }
        if (pending_space && !output.empty()) output.push_back(' ');
        pending_space = false;
        output.push_back(static_cast<char>(value));
    }
    return output;
}

SanoFrontendResult SanoFrontend::process(
    const std::string& text,
    const SanoPiperConfig& config
) const {
    SanoFrontendResult result;
    if (!loaded()) return result;
    if (config.language == "en") {
        // Reuse the established embedded English normalizer for numbers,
        // dates, currency, abbreviations and contractions. It has no lexicon
        // dependency; Sano still uses its own SNL1 pronunciation table.
        V2Frontend normalizer;
        result.normalized_text = normalizer.normalize(text);
        // The generated word list commonly contains "piper" and "lite" but
        // not this product name. Split it before lookup instead of invoking
        // the generic OOV letter-spelling fallback.
        replace_word_ci(result.normalized_text, "Piperlite", "piper lite");
    } else {
        result.normalized_text = normalize(text);
    }
    if (result.normalized_text.empty()) return result;

    std::vector<uint8_t> ids;
    bool pending_space = false;
    auto append_pending_space = [&]() {
        if (pending_space && !result.phoneme_ids.empty() &&
            config.space_id >= 0 && config.space_id <= 255) {
            result.phoneme_ids.push_back(static_cast<uint8_t>(config.space_id));
        }
        pending_space = false;
    };

    const std::string& value = result.normalized_text;
    for (size_t pos = 0; pos < value.size();) {
        const unsigned char current = static_cast<unsigned char>(value[pos]);
        if (std::isspace(current)) {
            pending_space = !result.phoneme_ids.empty();
            ++pos;
            continue;
        }

        const bool apostrophe_in_word =
            value[pos] == '\'' && pos > 0 && pos + 1 < value.size() &&
            ascii_word_byte(static_cast<unsigned char>(value[pos - 1])) &&
            ascii_word_byte(static_cast<unsigned char>(value[pos + 1]));
        if (ascii_word_byte(current) || apostrophe_in_word) {
            const size_t begin = pos;
            while (pos < value.size()) {
                const unsigned char byte = static_cast<unsigned char>(value[pos]);
                const bool apostrophe =
                    value[pos] == '\'' && pos > begin && pos + 1 < value.size() &&
                    ascii_word_byte(static_cast<unsigned char>(value[pos - 1])) &&
                    ascii_word_byte(static_cast<unsigned char>(value[pos + 1]));
                if (!ascii_word_byte(byte) && !apostrophe) break;
                ++pos;
            }
            const std::string_view word(value.data() + begin, pos - begin);
            if (lookup(word, ids) ||
                (config.language == "en" && spell_ascii_word(word, config, ids))) {
                append_pending_space();
                append_ids(result.phoneme_ids, ids);
            } else {
                result.oov_words.emplace_back(word);
                pending_space = !result.phoneme_ids.empty();
            }
            continue;
        }

        const int32_t punctuation = config.punctuation_id(value[pos]);
        if (punctuation >= 0 && punctuation <= 255) {
            // Piper preserves punctuation adjacent to the prior phoneme; a
            // whitespace separator is only materialized before the next word.
            pending_space = false;
            result.phoneme_ids.push_back(static_cast<uint8_t>(punctuation));
        }
        ++pos;
    }

    if (result.phoneme_ids.empty()) return result;
    const size_t framed_size = 3 + result.phoneme_ids.size() * 2;
    if (framed_size > static_cast<size_t>(config.duration_max_tokens)) {
        std::fprintf(stderr,
                     "[SanoFrontend] token budget exceeded: %zu > %d\n",
                     framed_size, config.duration_max_tokens);
        return result;
    }

    result.tokens.reserve(framed_size);
    result.tokens.push_back(SanoPiperConfig::kBosId);
    result.tokens.push_back(SanoPiperConfig::kPadId);
    for (uint8_t id : result.phoneme_ids) {
        result.tokens.push_back(id);
        result.tokens.push_back(SanoPiperConfig::kPadId);
    }
    result.tokens.push_back(SanoPiperConfig::kEosId);
    result.ok = true;
    return result;
}

} // namespace inflect
