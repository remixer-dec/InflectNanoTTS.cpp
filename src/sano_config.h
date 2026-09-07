#pragma once

#include "model_loader.h"

#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <string>

namespace inflect {

struct SanoPiperConfig {
    static constexpr int32_t kPadId = 0;
    static constexpr int32_t kBosId = 1;
    static constexpr int32_t kEosId = 2;
    static constexpr int32_t kSchwaFallbackId = 59;
    static constexpr int32_t kHopLength = 256;

    std::string voice;
    std::string language;
    std::string espeak_voice;
    std::string frontend_hash;

    int32_t sample_rate = 22050;
    int32_t hop_length = kHopLength;
    float duration_length_scale = 1.0f;

    int32_t duration_vocab = 0;
    int32_t duration_hidden = 0;
    int32_t duration_depth = 0;
    int32_t duration_kernel = 5;
    int32_t duration_max_tokens = 0;
    int32_t duration_max_duration = 0;

    int32_t acoustic_vocab = 0;
    int32_t acoustic_hidden = 0;
    int32_t acoustic_token_depth = 0;
    int32_t acoustic_depth = 0;
    int32_t acoustic_kernel = 5;
    int32_t acoustic_out_channels = 0;

    std::array<int32_t, 4> decoder_channels{{0, 0, 0, 0}};
    std::array<int32_t, 3> decoder_up_kernels{{16, 16, 8}};
    std::array<int32_t, 3> decoder_up_strides{{8, 8, 4}};
    std::array<int32_t, 3> decoder_up_paddings{{4, 4, 2}};
    int32_t decoder_pre_kernel = 7;
    int32_t decoder_post_kernel = 7;

    // bit 0/1/2 -> residual-bank branches using kernels 3/5/7.
    std::array<uint8_t, 3> decoder_branch_masks{{7, 7, 7}};

    int32_t post_filter_channels = 0;
    int32_t post_filter_layers = 0;
    int32_t post_filter_kernel = 9;
    float post_filter_scale = 0.0f;

    // Voice-specific punctuation ids copied from phoneme_id_map by the
    // converter.  Keeping these scalar avoids a heap map in the embedded
    // runtime; the full mapping is only needed offline when building SNL1.
    int32_t space_id = -1;
    int32_t apostrophe_id = -1;
    int32_t bang_id = -1;
    int32_t lparen_id = -1;
    int32_t rparen_id = -1;
    int32_t comma_id = -1;
    int32_t dash_id = -1;
    int32_t period_id = -1;
    int32_t colon_id = -1;
    int32_t semicolon_id = -1;
    int32_t question_id = -1;
    int32_t quote_id = -1;

    int32_t punctuation_id(char value) const {
        switch (value) {
            case '\'': return apostrophe_id;
            case '!': return bang_id;
            case '(': return lparen_id;
            case ')': return rparen_id;
            case ',': return comma_id;
            case '-': return dash_id;
            case '.': return period_id;
            case ':': return colon_id;
            case ';': return semicolon_id;
            case '?': return question_id;
            case '"': return quote_id;
            default: return -1;
        }
    }

    int32_t clamp_id(int32_t id, int32_t vocab) const {
        if (id >= 0 && id < vocab) return id;
        return kSchwaFallbackId < vocab ? kSchwaFallbackId : kPadId;
    }

    bool branch_enabled(int stage, int branch) const {
        return stage >= 0 && stage < 3 && branch >= 0 && branch < 3 &&
               (decoder_branch_masks[stage] & (1u << branch)) != 0;
    }

    int branch_count(int stage) const {
        if (stage < 0 || stage >= 3) return 0;
        uint8_t mask = decoder_branch_masks[stage];
        return static_cast<int>(mask & 1u) +
               static_cast<int>((mask >> 1) & 1u) +
               static_cast<int>((mask >> 2) & 1u);
    }

    bool load(const ModelLoader& loader) {
        const std::string architecture =
            loader.get_string("general.architecture", "");
        if (architecture != "sanotts-piperlite") {
            std::fprintf(stderr,
                         "[SanoConfig] incompatible architecture: %s\n",
                         architecture.c_str());
            return false;
        }
        const int32_t layout_version =
            loader.get_i32("sanotts.layout_version", 0);
        if (layout_version != 1) {
            std::fprintf(stderr,
                         "[SanoConfig] unsupported layout version: %d\n",
                         layout_version);
            return false;
        }

        voice = loader.get_string("sanotts.voice", "");
        language = loader.get_string("sanotts.language", "");
        espeak_voice = loader.get_string("sanotts.espeak_voice", "");
        frontend_hash = loader.get_string("sanotts.frontend.map_hash", "");
        sample_rate = loader.get_i32("sanotts.sample_rate", 22050);
        hop_length = loader.get_i32("sanotts.hop_length", 0);
        duration_length_scale =
            loader.get_f32("sanotts.duration_length_scale", 1.0f);

        duration_vocab = loader.get_i32("sanotts.duration.vocab_size", 0);
        duration_hidden = loader.get_i32("sanotts.duration.hidden", 0);
        duration_depth = loader.get_i32("sanotts.duration.depth", 0);
        duration_kernel = loader.get_i32("sanotts.duration.kernel", 0);
        duration_max_tokens =
            loader.get_i32("sanotts.duration.max_tokens", 0);
        duration_max_duration =
            loader.get_i32("sanotts.duration.max_duration", 0);

        acoustic_vocab = loader.get_i32("sanotts.acoustic.vocab_size", 0);
        acoustic_hidden = loader.get_i32("sanotts.acoustic.hidden", 0);
        acoustic_token_depth =
            loader.get_i32("sanotts.acoustic.token_depth", 0);
        acoustic_depth = loader.get_i32("sanotts.acoustic.depth", 0);
        acoustic_kernel = loader.get_i32("sanotts.acoustic.kernel", 0);
        acoustic_out_channels =
            loader.get_i32("sanotts.acoustic.out_channels", 0);

        for (int index = 0; index < 4; ++index) {
            decoder_channels[index] = loader.get_i32(
                "sanotts.decoder.channel" + std::to_string(index), 0);
        }
        decoder_pre_kernel =
            loader.get_i32("sanotts.decoder.pre_kernel", 7);
        decoder_post_kernel =
            loader.get_i32("sanotts.decoder.post_kernel", 7);
        for (int stage = 0; stage < 3; ++stage) {
            decoder_up_kernels[stage] = loader.get_i32(
                "sanotts.decoder.up" + std::to_string(stage) + "_kernel",
                decoder_up_kernels[stage]);
            decoder_up_strides[stage] = loader.get_i32(
                "sanotts.decoder.up" + std::to_string(stage) + "_stride",
                decoder_up_strides[stage]);
            decoder_up_paddings[stage] = loader.get_i32(
                "sanotts.decoder.up" + std::to_string(stage) + "_padding",
                decoder_up_paddings[stage]);
            decoder_branch_masks[stage] = static_cast<uint8_t>(loader.get_i32(
                "sanotts.decoder.stage" + std::to_string(stage) +
                    "_branch_mask",
                0));
        }

        post_filter_channels =
            loader.get_i32("sanotts.decoder.post_filter_channels", 0);
        post_filter_layers =
            loader.get_i32("sanotts.decoder.post_filter_layers", 0);
        post_filter_kernel =
            loader.get_i32("sanotts.decoder.post_filter_kernel", 9);
        post_filter_scale =
            loader.get_f32("sanotts.decoder.post_filter_scale", 0.0f);

        space_id = loader.get_i32("sanotts.frontend.space_id", -1);
        apostrophe_id =
            loader.get_i32("sanotts.frontend.punc.apostrophe", -1);
        bang_id = loader.get_i32("sanotts.frontend.punc.bang", -1);
        lparen_id = loader.get_i32("sanotts.frontend.punc.lparen", -1);
        rparen_id = loader.get_i32("sanotts.frontend.punc.rparen", -1);
        comma_id = loader.get_i32("sanotts.frontend.punc.comma", -1);
        dash_id = loader.get_i32("sanotts.frontend.punc.dash", -1);
        period_id = loader.get_i32("sanotts.frontend.punc.period", -1);
        colon_id = loader.get_i32("sanotts.frontend.punc.colon", -1);
        semicolon_id =
            loader.get_i32("sanotts.frontend.punc.semicolon", -1);
        question_id = loader.get_i32("sanotts.frontend.punc.question", -1);
        quote_id = loader.get_i32("sanotts.frontend.punc.quote", -1);

        return validate();
    }

    bool validate() const {
        if (sample_rate <= 0 || hop_length != kHopLength ||
            !(duration_length_scale > 0.0f) || !std::isfinite(duration_length_scale) ||
            duration_vocab <= 0 || duration_hidden <= 0 || duration_depth <= 0 ||
            duration_kernel <= 0 || (duration_kernel & 1) == 0 ||
            duration_max_tokens <= 0 || duration_max_duration <= 0 ||
            acoustic_vocab <= 0 || acoustic_hidden <= 0 ||
            acoustic_token_depth <= 0 || acoustic_depth <= 0 ||
            acoustic_kernel <= 0 || (acoustic_kernel & 1) == 0 ||
            acoustic_out_channels <= 0) {
            std::fprintf(stderr, "[SanoConfig] invalid piperlite front geometry\n");
            return false;
        }
        if (frontend_hash.size() != 64 || voice.empty() || language.empty()) {
            std::fprintf(stderr, "[SanoConfig] invalid frontend metadata\n");
            return false;
        }
        for (int32_t channel : decoder_channels) {
            if (channel <= 0) {
                std::fprintf(stderr, "[SanoConfig] invalid decoder channels\n");
                return false;
            }
        }
        for (int stage = 0; stage < 3; ++stage) {
            const uint8_t mask = decoder_branch_masks[stage];
            if ((mask & 0x07u) == 0 || (mask & ~0x07u) != 0 ||
                decoder_up_kernels[stage] <= 0 ||
                decoder_up_strides[stage] <= 0 ||
                decoder_up_paddings[stage] < 0) {
                std::fprintf(stderr, "[SanoConfig] invalid decoder stage geometry\n");
                return false;
            }
            if (static_cast<int64_t>(decoder_up_kernels[stage]) -
                    2LL * decoder_up_paddings[stage] != decoder_up_strides[stage]) {
                std::fprintf(stderr, "[SanoConfig] upsample must preserve stride phase\n");
                return false;
            }
        }
        if (static_cast<int64_t>(decoder_up_strides[0]) *
                decoder_up_strides[1] * decoder_up_strides[2] != kHopLength) return false;
        if (decoder_pre_kernel <= 0 || (decoder_pre_kernel & 1) == 0 ||
            decoder_post_kernel <= 0 || (decoder_post_kernel & 1) == 0) {
            std::fprintf(stderr, "[SanoConfig] invalid decoder edge kernels\n");
            return false;
        }
        if ((post_filter_channels == 0) != (post_filter_layers == 0) ||
            post_filter_channels < 0 || post_filter_layers < 0 ||
            (post_filter_channels > 0 &&
             (post_filter_kernel <= 0 || (post_filter_kernel & 1) == 0))) {
            std::fprintf(stderr, "[SanoConfig] invalid post-filter geometry\n");
            return false;
        }
        return true;
    }
};

} // namespace inflect
