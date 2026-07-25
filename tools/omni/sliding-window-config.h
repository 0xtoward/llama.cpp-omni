#pragma once

#include <charconv>
#include <cstdint>
#include <string>
#include <utility>

namespace omni::sliding_window {

// Leave enough room for the model's response and duplex control tokens. This
// is deliberately fixed for the direct-test interface so an enabled window
// cannot be configured right at the KV capacity boundary.
inline constexpr int32_t k_context_reserve_tokens = 256;

inline bool parse_positive_int(
        const char * name,
        const char * value,
        int & result,
        std::string & error) {
    if (value == nullptr) {
        return true;
    }

    int parsed = 0;
    const char * const end = value + std::char_traits<char>::length(value);
    const auto conversion = std::from_chars(value, end, parsed);
    if (conversion.ec != std::errc{} || conversion.ptr != end || parsed <= 0) {
        error = std::string(name) + " must be a positive base-10 integer";
        return false;
    }
    result = parsed;
    return true;
}

// Applies the direct-test environment overrides to the existing defaults.
// With no override present the configuration is left byte-for-byte unchanged,
// including legacy mode names that may still be set through the server API.
inline bool parse_env(
        const char * mode_env,
        const char * high_env,
        const char * low_env,
        int32_t n_ctx,
        std::string & mode,
        int & high_water_tokens,
        int & low_water_tokens,
        std::string & error) {
    if (mode_env == nullptr && high_env == nullptr && low_env == nullptr) {
        return true;
    }

    std::string parsed_mode = mode;
    int parsed_high = high_water_tokens;
    int parsed_low = low_water_tokens;

    if (mode_env != nullptr) {
        parsed_mode = mode_env;
        if (parsed_mode != "off" &&
            parsed_mode != "unit" &&
            parsed_mode != "turn") {
            error = "OMNI_SLIDING_WINDOW_MODE must be off, unit, or turn";
            return false;
        }
    }
    if (!parse_positive_int(
                "OMNI_SLIDING_WINDOW_HIGH", high_env, parsed_high, error) ||
        !parse_positive_int(
                "OMNI_SLIDING_WINDOW_LOW", low_env, parsed_low, error)) {
        return false;
    }
    if (parsed_high <= parsed_low) {
        error = "sliding-window thresholds must satisfy high > low > 0";
        return false;
    }

    // Disabled watermarks are inert. Keeping this exception lets an explicit
    // MODE=off preserve the historical 4000/3500 defaults at n_ctx=4096.
    if (parsed_mode != "off") {
        if (n_ctx <= k_context_reserve_tokens) {
            error =
                    "sliding window requires n_ctx > reserve (" +
                    std::to_string(n_ctx) + " <= " +
                    std::to_string(k_context_reserve_tokens) + ")";
            return false;
        }
        const int32_t high_limit = n_ctx - k_context_reserve_tokens;
        if (parsed_high >= high_limit) {
            error =
                    "OMNI_SLIDING_WINDOW_HIGH must be < n_ctx - reserve (" +
                    std::to_string(parsed_high) + " >= " +
                    std::to_string(high_limit) + "; n_ctx=" +
                    std::to_string(n_ctx) + ", reserve=" +
                    std::to_string(k_context_reserve_tokens) + ")";
            return false;
        }
    }

    mode = std::move(parsed_mode);
    high_water_tokens = parsed_high;
    low_water_tokens = parsed_low;
    return true;
}

}  // namespace omni::sliding_window
