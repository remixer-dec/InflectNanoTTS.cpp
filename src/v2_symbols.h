#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace inflect {
namespace v2 {

constexpr uint8_t kBlankId = 0;
constexpr uint8_t kSpaceId = 16;
constexpr size_t kSymbolCount = 178;
constexpr const char* kSymbolHashHex =
    "3e81afeec2d0906de3d7acf2214d32fbc066be8218d2edafe355255391ea92f7";

// SHA-256 of the UTF-8 bytes of the concatenated released symbol list.
constexpr std::array<uint8_t, 32> kSymbolHash = {
    0x3e, 0x81, 0xaf, 0xee, 0xc2, 0xd0, 0x90, 0x6d,
    0xe3, 0xd7, 0xac, 0xf2, 0x21, 0x4d, 0x32, 0xfb,
    0xc0, 0x66, 0xbe, 0x82, 0x18, 0xd2, 0xed, 0xaf,
    0xe3, 0x55, 0x25, 0x53, 0x91, 0xea, 0x92, 0xf7,
};

const std::vector<std::string_view>& symbols();
int symbol_id(std::string_view utf8_codepoint);
std::vector<uint8_t> intersperse_blanks(const std::vector<uint8_t>& tokens);

} // namespace v2
} // namespace inflect
