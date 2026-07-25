#include "../sliding-window-config.h"

#include <cstdlib>
#include <string>

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

    return 0;
}
