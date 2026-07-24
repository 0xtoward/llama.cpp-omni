#include "token2wav/token2wav-hift-policy.h"

#include <iostream>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    using namespace omni::flow;

    {
        const auto config = parse_hift_runner_config(nullptr, nullptr);
        CHECK(config);
        CHECK(config.mode == hift_runner_mode::ephemeral);
        CHECK(config.plan_cache_capacity == 6);
    }
    {
        const auto config = parse_hift_runner_config("persistent", "3");
        CHECK(config);
        CHECK(config.mode == hift_runner_mode::persistent);
        CHECK(config.plan_cache_capacity == 3);
    }
    {
        const auto config = parse_hift_runner_config("persistent_graph", "64");
        CHECK(config);
        CHECK(config.mode == hift_runner_mode::persistent_graph);
        CHECK(config.plan_cache_capacity == 64);
    }
    for (const char * invalid : { "", "on", "PERSISTENT", "graph" }) {
        CHECK(!parse_hift_runner_config(invalid, nullptr));
    }
    for (const char * invalid : { "", "0", "-1", "65", "1x" }) {
        CHECK(!parse_hift_runner_config("persistent", invalid));
    }
    {
        const hift_plan_key first{hift_plan_phase_for(false, 0), 28, 0};
        const hift_plan_key steady{hift_plan_phase_for(false, 3840), 33, 3840};
        const hift_plan_key final{hift_plan_phase_for(true, 3840), 33, 3840};
        CHECK((first == hift_plan_key{hift_plan_phase::first, 28, 0}));
        CHECK(!(steady == final));
        CHECK(hift_plan_key_hash{}(steady) != hift_plan_key_hash{}(final));
    }
    {
        const hift_buffer_span input{0x1000, 0x100};
        CHECK(hift_buffer_spans_overlap(input, {0x1080, 0x20}));
        CHECK(hift_buffer_spans_overlap({0x1080, 0x20}, input));
        CHECK(!hift_buffer_spans_overlap(input, {0x1100, 0x20}));
        CHECK(!hift_buffer_spans_overlap({0x0f00, 0x100}, input));
        CHECK(!hift_buffer_spans_overlap(input, {0x1080, 0}));
    }

    std::cout << "HiFT runner policy tests passed\n";
    return 0;
}

#undef CHECK
