#pragma once

#include <cstddef>
#include <limits>

struct omni_sampling_policy {
    bool duplex_mode = false;
    float listen_prob_scale = 1.0f;
    float length_penalty = 1.0f;
    int special_token_listen = -1;
    int special_token_tts_pad = -1;
    int special_token_turn_eos = -1;
    int special_token_tts_eos = -1;
};

inline bool omni_backend_sampling_controls_supported(
        bool backend_sampling_active,
        float length_penalty,
        float listen_prob_scale) {
    return !backend_sampling_active ||
           (length_penalty == 1.0f && listen_prob_scale == 1.0f);
}

inline void omni_apply_host_sampling_policy(
        float * logits,
        size_t logits_count,
        const omni_sampling_policy & policy) {
    if (logits == nullptr) {
        return;
    }

    const auto valid_token = [logits_count](int token) {
        return token >= 0 && static_cast<size_t>(token) < logits_count;
    };
    const auto apply_length_penalty = [&](int token) {
        if (policy.length_penalty == 1.0f || !valid_token(token)) {
            return;
        }
        float & logit = logits[token];
        logit = logit > 0.0f ? logit / policy.length_penalty
                             : logit * policy.length_penalty;
    };

    if (policy.duplex_mode) {
        if (valid_token(policy.special_token_listen)) {
            logits[policy.special_token_listen] +=
                (policy.listen_prob_scale - 1.0f) * 2.0f;
        }
        if (valid_token(policy.special_token_tts_pad)) {
            logits[policy.special_token_tts_pad] =
                -std::numeric_limits<float>::infinity();
        }
        apply_length_penalty(policy.special_token_turn_eos);
    } else {
        apply_length_penalty(policy.special_token_tts_eos);
    }
}
