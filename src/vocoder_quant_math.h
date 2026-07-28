#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace inflect::vocoder_quant {

inline uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float float_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint16_t fp32_to_fp16_bits(float value) {
    constexpr float kScaleToInf = 0x1.0p+112f;
    constexpr float kScaleToZero = 0x1.0p-110f;
    float base =
        (std::fabs(value) * kScaleToInf) * kScaleToZero;

    const uint32_t bits = float_bits(value);
    const uint32_t shifted = bits + bits;
    const uint32_t sign = bits & 0x80000000U;
    uint32_t bias = shifted & 0xff000000U;
    if (bias < 0x71000000U) {
        bias = 0x71000000U;
    }

    base =
        float_from_bits((bias >> 1) + 0x07800000U) + base;
    const uint32_t base_bits = float_bits(base);
    const uint32_t exponent =
        (base_bits >> 13) & 0x00007c00U;
    const uint32_t mantissa = base_bits & 0x00000fffU;
    const uint32_t non_sign = exponent + mantissa;
    return static_cast<uint16_t>(
        (sign >> 16) |
        (shifted > 0xff000000U ? 0x7e00U : non_sign));
}

inline float fp16_bits_to_fp32(uint16_t value) {
    const uint32_t word = static_cast<uint32_t>(value) << 16;
    const uint32_t sign = word & 0x80000000U;
    const uint32_t doubled = word + word;
    constexpr float kExponentScale = 0x1.0p-112f;
    const float normalized =
        float_from_bits(
            (doubled >> 4) + (0xe0U << 23)) *
        kExponentScale;
    const float denormalized =
        float_from_bits(
            (doubled >> 17) | (126U << 23)) -
        0.5f;
    const uint32_t result =
        sign |
        float_bits(
            doubled < (1U << 27)
                ? denormalized
                : normalized);
    return float_from_bits(result);
}

inline float cache_scale_fp16(float scale) {
    return fp16_bits_to_fp32(fp32_to_fp16_bits(scale));
}

inline int round_half_away_from_zero(float value) {
    const int truncated = static_cast<int>(value);
    const float remainder =
        value - static_cast<float>(truncated);
    return truncated +
           static_cast<int>(remainder >= 0.5f) -
           static_cast<int>(remainder <= -0.5f);
}

// Exact for finite IEEE-754 binary32 values whose magnitude is below 128.
// This is a benchmark candidate for targets where reconstructing the
// fractional float after truncation is expensive.
inline int round_half_away_from_zero_bits_bounded(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude_bits = bits & 0x7fffffffU;
    const uint32_t active_mask =
        0U - static_cast<uint32_t>(
                 magnitude_bits >= 0x3f000000U);
    const uint32_t exponent =
        (magnitude_bits >> 23) & 0xffU;
    const uint32_t safe_exponent =
        (exponent & active_mask) |
        (126U & ~active_mask);
    const uint32_t shift = 150U - safe_exponent;
    const uint32_t significand =
        (magnitude_bits & 0x007fffffU) | 0x00800000U;
    const uint32_t rounded_magnitude =
        ((significand + (1U << (shift - 1U))) >> shift) &
        active_mask;
    const int sign_mask =
        -static_cast<int>(bits >> 31);
    return (static_cast<int>(rounded_magnitude) ^ sign_mask) -
           sign_mask;
}

// Exact for finite IEEE-754 binary32 values in the Q8 conversion range
// [-127, 127]. The guard prevents a value immediately below 0.5 from
// rounding up during the addition. Above that boundary, adding a
// sign-matched 0.5 and truncating implements half-away-from-zero.
inline int round_half_away_from_zero_add_bounded(float value) {
    const uint32_t bits = float_bits(value);
    if ((bits & 0x7fffffffU) < 0x3f000000U) {
        return 0;
    }
    const float signed_half = float_from_bits(
        (bits & 0x80000000U) | 0x3f000000U);
    return static_cast<int>(value + signed_half);
}

inline int8_t round_s8(float value) {
    const int rounded = round_half_away_from_zero(value);
    const int clamped = rounded < -127
        ? -127
        : (rounded > 127 ? 127 : rounded);
    return static_cast<int8_t>(clamped);
}

inline float absolute_max_32(const float* values) {
    float maximum0 = 0.0f;
    float maximum1 = 0.0f;
    float maximum2 = 0.0f;
    float maximum3 = 0.0f;
    float maximum4 = 0.0f;
    float maximum5 = 0.0f;
    float maximum6 = 0.0f;
    float maximum7 = 0.0f;

    for (int index = 0; index < 32; index += 8) {
        maximum0 = std::max(maximum0, std::fabs(values[index]));
        maximum1 = std::max(maximum1, std::fabs(values[index + 1]));
        maximum2 = std::max(maximum2, std::fabs(values[index + 2]));
        maximum3 = std::max(maximum3, std::fabs(values[index + 3]));
        maximum4 = std::max(maximum4, std::fabs(values[index + 4]));
        maximum5 = std::max(maximum5, std::fabs(values[index + 5]));
        maximum6 = std::max(maximum6, std::fabs(values[index + 6]));
        maximum7 = std::max(maximum7, std::fabs(values[index + 7]));
    }

    const float maximum01 = std::max(maximum0, maximum1);
    const float maximum23 = std::max(maximum2, maximum3);
    const float maximum45 = std::max(maximum4, maximum5);
    const float maximum67 = std::max(maximum6, maximum7);
    return std::max(
        std::max(maximum01, maximum23),
        std::max(maximum45, maximum67));
}

inline void quantize_s8_32(
    const float* source,
    float inverse,
    int8_t* destination
) {
    for (int index = 0; index < 32; index += 8) {
        destination[index] = static_cast<int8_t>(
            round_half_away_from_zero(source[index] * inverse));
        destination[index + 1] =
            static_cast<int8_t>(round_half_away_from_zero(
                source[index + 1] * inverse));
        destination[index + 2] =
            static_cast<int8_t>(round_half_away_from_zero(
                source[index + 2] * inverse));
        destination[index + 3] =
            static_cast<int8_t>(round_half_away_from_zero(
                source[index + 3] * inverse));
        destination[index + 4] =
            static_cast<int8_t>(round_half_away_from_zero(
                source[index + 4] * inverse));
        destination[index + 5] =
            static_cast<int8_t>(round_half_away_from_zero(
                source[index + 5] * inverse));
        destination[index + 6] =
            static_cast<int8_t>(round_half_away_from_zero(
                source[index + 6] * inverse));
        destination[index + 7] =
            static_cast<int8_t>(round_half_away_from_zero(
                source[index + 7] * inverse));
    }
}

inline float quantize_q8_32_bits(
    const float* source,
    int8_t* destination
) {
    const float maximum = absolute_max_32(source);
    const float scale = maximum / 127.0f;
    const float inverse =
        scale != 0.0f ? 1.0f / scale : 0.0f;
    for (int index = 0; index < 32; ++index) {
        destination[index] = static_cast<int8_t>(
            round_half_away_from_zero_bits_bounded(
                source[index] * inverse));
    }
    return cache_scale_fp16(scale);
}

inline void unpack_q4_0_32_unsigned_swar(
    const uint8_t* source,
    uint8_t* destination
) {
    constexpr uint16_t kNibbleMask = 0x0f0fU;
    for (int index = 0; index < 16; index += 2) {
        uint16_t packed;
        std::memcpy(&packed, source + index, sizeof(packed));
        const uint16_t low = packed & kNibbleMask;
        const uint16_t high =
            static_cast<uint16_t>((packed >> 4) & kNibbleMask);
        std::memcpy(destination + index, &low, sizeof(low));
        std::memcpy(
            destination + 16 + index, &high, sizeof(high));
    }
}

inline void unpack_q4_0_32_swar(
    const uint8_t* source,
    int8_t* destination
) {
    constexpr uint16_t kNibbleMask = 0x0f0fU;
    constexpr uint16_t kSignedBias = 0x7878U;
    constexpr uint16_t kSignBits = 0x8080U;
    for (int index = 0; index < 16; index += 2) {
        uint16_t packed;
        std::memcpy(&packed, source + index, sizeof(packed));
        uint16_t low = packed & kNibbleMask;
        uint16_t high =
            static_cast<uint16_t>((packed >> 4) & kNibbleMask);
        low = static_cast<uint16_t>(
            (low + kSignedBias) ^ kSignBits);
        high = static_cast<uint16_t>(
            (high + kSignedBias) ^ kSignBits);
        std::memcpy(destination + index, &low, sizeof(low));
        std::memcpy(
            destination + 16 + index, &high, sizeof(high));
    }
}

} // namespace inflect::vocoder_quant
