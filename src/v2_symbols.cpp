#include "v2_symbols.h"

#include <unordered_map>

namespace inflect {
namespace v2 {

const std::vector<std::string_view>& symbols() {
    // Keep this byte-for-byte aligned with runtime/text/symbols.py at the
    // pinned Inflect-Nano-v2 revision. Apostrophe intentionally occurs twice;
    // Python's dict comprehension resolves it to the later ID (176).
    static const std::vector<std::string_view> value = {
        "_", ";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"",
        "«", "»", "“", "”", " ",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
        "Y", "Z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
        "y", "z",
        "ɑ", "ɐ", "ɒ", "æ", "ɓ", "ʙ", "β", "ɔ", "ɕ", "ç", "ɗ", "ɖ",
        "ð", "ʤ", "ə", "ɘ", "ɚ", "ɛ", "ɜ", "ɝ", "ɞ", "ɟ", "ʄ", "ɡ",
        "ɠ", "ɢ", "ʛ", "ɦ", "ɧ", "ħ", "ɥ", "ʜ", "ɨ", "ɪ", "ʝ", "ɭ",
        "ɬ", "ɫ", "ɮ", "ʟ", "ɱ", "ɯ", "ɰ", "ŋ", "ɳ", "ɲ", "ɴ", "ø",
        "ɵ", "ɸ", "θ", "œ", "ɶ", "ʘ", "ɹ", "ɺ", "ɾ", "ɻ", "ʀ", "ʁ",
        "ɽ", "ʂ", "ʃ", "ʈ", "ʧ", "ʉ", "ʊ", "ʋ", "ⱱ", "ʌ", "ɣ", "ɤ",
        "ʍ", "χ", "ʎ", "ʏ", "ʑ", "ʐ", "ʒ", "ʔ", "ʡ", "ʕ", "ʢ", "ǀ",
        "ǁ", "ǂ", "ǃ", "ˈ", "ˌ", "ː", "ˑ", "ʼ", "ʴ", "ʰ", "ʱ", "ʲ",
        "ʷ", "ˠ", "ˤ", "˞", "↓", "↑", "→", "↗", "↘", "'", "̩", "'", "ᵻ",
    };
    return value;
}

int symbol_id(std::string_view utf8_codepoint) {
    static const auto ids = [] {
        std::unordered_map<std::string_view, int> result;
        const auto& table = symbols();
        for (size_t i = 0; i < table.size(); ++i) {
            result[table[i]] = static_cast<int>(i);
        }
        return result;
    }();
    const auto it = ids.find(utf8_codepoint);
    return it == ids.end() ? -1 : it->second;
}

std::vector<uint8_t> intersperse_blanks(const std::vector<uint8_t>& tokens) {
    std::vector<uint8_t> result;
    result.reserve(tokens.size() * 2 + 1);
    result.push_back(kBlankId);
    for (uint8_t token : tokens) {
        result.push_back(token);
        result.push_back(kBlankId);
    }
    return result;
}

} // namespace v2
} // namespace inflect
