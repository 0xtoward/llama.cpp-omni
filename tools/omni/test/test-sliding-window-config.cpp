#include "../sliding-window-config.h"

#include <cstdlib>
#include <string>
#include <vector>

static void check(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static bool parse(
        const char * mode_env,
        const char * high_env,
        const char * low_env,
        int32_t n_ctx,
        std::string & mode,
        int & high,
        int & low,
        std::string & error) {
    error.clear();
    return omni::sliding_window::parse_env(
            mode_env, high_env, low_env, n_ctx, mode, high, low, error);
}

int main() {
    {
        std::string mode = "context";
        int high = 4000;
        int low = 3500;
        std::string error;
        check(parse(nullptr, nullptr, nullptr, 4096, mode, high, low, error));
        check(mode == "context" && high == 4000 && low == 3500);
    }
    {
        std::string mode = "off";
        int high = 4000;
        int low = 3500;
        std::string error;
        check(parse("off", nullptr, nullptr, 4096, mode, high, low, error));
        check(mode == "off" && high == 4000 && low == 3500);
    }
    {
        std::string mode = "off";
        int high = 4000;
        int low = 3500;
        std::string error;
        check(parse("unit", "3500", "3000", 4096, mode, high, low, error));
        check(mode == "unit" && high == 3500 && low == 3000);
        check(parse("turn", "3000", "2000", 4096, mode, high, low, error));
        check(mode == "turn" && high == 3000 && low == 2000);
    }
    {
        std::string mode = "off";
        int high = 4000;
        int low = 3500;
        std::string error;
        check(!parse("basic", "3500", "3000", 4096, mode, high, low, error));
        check(error.find("off, unit, or turn") != std::string::npos);
        check(mode == "off" && high == 4000 && low == 3500);
    }
    {
        std::string mode = "off";
        int high = 4000;
        int low = 3500;
        std::string error;
        check(!parse("unit", "3000", "3000", 4096, mode, high, low, error));
        check(error.find("high > low > 0") != std::string::npos);
        check(!parse("unit", "3000", "0", 4096, mode, high, low, error));
        check(error.find("positive base-10 integer") != std::string::npos);
        check(!parse("unit", " 3000", "2000", 4096, mode, high, low, error));
        check(!parse("unit", "+3000", "2000", 4096, mode, high, low, error));
    }
    {
        std::string mode = "off";
        int high = 4000;
        int low = 3500;
        std::string error;
        check(!parse("unit", "3840", "3000", 4096, mode, high, low, error));
        check(error.find("n_ctx - reserve") != std::string::npos);
        check(error.find("reserve=256") != std::string::npos);
        check(!parse("unit", "100", "50", 256, mode, high, low, error));
        check(error.find("n_ctx > reserve") != std::string::npos);
    }
    {
        struct unit {
            int unit_id;
            int length;
            std::string type;
            int turn_id;
        };

        // The current-turn fallback batches the oldest units only until the
        // low-water target is reached. System metadata remains protected.
        const std::vector<unit> units = {
            {0, 1200, "system", 0},
            {1,   50, "audio",  7},
            {2,   40, "response", 7},
            {3,   30, "audio",  7},
        };
        const auto batch =
                omni::sliding_window::plan_unit_fallback_batch(
                        units, 7, -1, 4190, 4100);
        check(batch.token_count == 90);
        check(batch.unit_ids == std::vector<int>({1, 2}));

        // Discrete units may cross below the target, but the newest unit is
        // left alone once enough tokens have been accumulated.
        const auto overshoot =
                omni::sliding_window::plan_unit_fallback_batch(
                        units, 7, -1, 4160, 4100);
        check(overshoot.token_count == 90);
        check(overshoot.unit_ids == std::vector<int>({1, 2}));

        // A pending unit is a hard boundary even if more tokens are needed.
        const auto pending =
                omni::sliding_window::plan_unit_fallback_batch(
                        units, 7, 2, 4300, 4100);
        check(pending.token_count == 50);
        check(pending.unit_ids == std::vector<int>({1}));

        // If an older completed turn survived the turn-first pass, fail
        // closed instead of silently crossing a turn boundary.
        std::vector<unit> stale_turn = units;
        stale_turn[1].turn_id = 6;
        const auto stale =
                omni::sliding_window::plan_unit_fallback_batch(
                        stale_turn, 7, -1, 4300, 4100);
        check(stale.token_count == 0);
        check(stale.unit_ids.empty());

        const auto below_low =
                omni::sliding_window::plan_unit_fallback_batch(
                        units, 7, -1, 4099, 4100);
        check(below_low.token_count == 0);
        check(below_low.unit_ids.empty());

        // Regression for the long-soak failure mode: 98 oldest units are
        // planned as one compaction instead of 98 KV rm+shift operations.
        std::vector<unit> long_turn;
        for (int i = 0; i < 100; ++i) {
            long_turn.push_back({i, 50, "audio", 11});
        }
        const auto long_batch =
                omni::sliding_window::plan_unit_fallback_batch(
                        long_turn, 11, -1, 9000, 4100);
        check(long_batch.token_count == 4900);
        check(long_batch.unit_ids.size() == 98);
        check(long_batch.unit_ids.front() == 0);
        check(long_batch.unit_ids.back() == 97);
    }

    return 0;
}
