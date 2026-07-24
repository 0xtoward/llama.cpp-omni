#include "sampling-policy.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    assert(omni_backend_sampling_controls_supported(false, 1.1f, 6.0f));
    assert(omni_backend_sampling_controls_supported(true, 1.0f, 1.0f));
    assert(!omni_backend_sampling_controls_supported(true, 1.1f, 1.0f));
    assert(!omni_backend_sampling_controls_supported(true, 1.0f, 6.0f));

    std::vector<float> logits = {0.0f, 1.0f, 7.0f, 2.0f};
    omni_sampling_policy duplex;
    duplex.duplex_mode = true;
    duplex.listen_prob_scale = 6.0f;
    duplex.length_penalty = 2.0f;
    duplex.special_token_listen = 1;
    duplex.special_token_tts_pad = 2;
    duplex.special_token_turn_eos = 3;
    omni_apply_host_sampling_policy(logits.data(), logits.size(), duplex);
    assert(logits[1] == 11.0f);
    assert(std::isinf(logits[2]) && logits[2] < 0.0f);
    assert(logits[3] == 1.0f);

    std::vector<float> single = {0.0f, -3.0f};
    omni_sampling_policy turn_based;
    turn_based.length_penalty = 2.0f;
    turn_based.special_token_tts_eos = 1;
    omni_apply_host_sampling_policy(single.data(), single.size(), turn_based);
    assert(single[1] == -6.0f);

    omni_sampling_policy invalid;
    invalid.duplex_mode = true;
    invalid.special_token_listen = 99;
    invalid.special_token_tts_pad = -1;
    invalid.special_token_turn_eos = 100;
    omni_apply_host_sampling_policy(single.data(), single.size(), invalid);
    return 0;
}
