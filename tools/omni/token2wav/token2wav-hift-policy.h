#pragma once

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>

namespace omni::flow {

enum class hift_runner_mode {
    ephemeral,
    persistent,
    persistent_graph,
};

enum class hift_plan_phase : uint8_t {
    first,
    steady,
    final,
};

struct hift_plan_key {
    hift_plan_phase phase = hift_plan_phase::steady;
    int64_t         t_mel = 0;
    int64_t         tc = 0;

    bool operator==(const hift_plan_key & other) const {
        return phase == other.phase && t_mel == other.t_mel && tc == other.tc;
    }
};

struct hift_plan_key_hash {
    size_t operator()(const hift_plan_key & value) const {
        size_t hash = static_cast<size_t>(value.phase);
        hash ^= std::hash<int64_t>{}(value.t_mel) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int64_t>{}(value.tc) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct hift_buffer_span {
    uintptr_t address = 0;
    size_t    bytes = 0;
};

inline bool hift_buffer_spans_overlap(
        hift_buffer_span lhs,
        hift_buffer_span rhs) {
    if (lhs.bytes == 0 || rhs.bytes == 0) {
        return false;
    }
    if (lhs.address <= rhs.address) {
        return rhs.address - lhs.address < lhs.bytes;
    }
    return lhs.address - rhs.address < rhs.bytes;
}

inline hift_plan_phase hift_plan_phase_for(bool is_final, int64_t tc) {
    return is_final ? hift_plan_phase::final
                    : (tc == 0 ? hift_plan_phase::first : hift_plan_phase::steady);
}

struct hift_runner_config {
    hift_runner_mode mode = hift_runner_mode::ephemeral;
    size_t           plan_cache_capacity = 6;
    std::string      error;

    explicit operator bool() const {
        return error.empty();
    }
};

inline const char * hift_runner_mode_name(hift_runner_mode mode) {
    switch (mode) {
        case hift_runner_mode::ephemeral:        return "ephemeral";
        case hift_runner_mode::persistent:       return "persistent";
        case hift_runner_mode::persistent_graph: return "persistent_graph";
    }
    return "invalid";
}

inline hift_runner_config parse_hift_runner_config(
        const char * mode_raw,
        const char * capacity_raw) {
    hift_runner_config result;

    const std::string mode = mode_raw == nullptr ? "ephemeral" : mode_raw;
    if (mode == "ephemeral") {
        result.mode = hift_runner_mode::ephemeral;
    } else if (mode == "persistent") {
        result.mode = hift_runner_mode::persistent;
    } else if (mode == "persistent_graph") {
        result.mode = hift_runner_mode::persistent_graph;
    } else {
        result.error = "OMNI_HIFT_RUNNER must be ephemeral|persistent|persistent_graph";
        return result;
    }

    if (capacity_raw != nullptr) {
        if (*capacity_raw == '\0') {
            result.error = "OMNI_HIFT_PLAN_CACHE_CAPACITY must be an integer in [1, 64]";
            return result;
        }
        char * end = nullptr;
        errno = 0;
        const long parsed = std::strtol(capacity_raw, &end, 10);
        if (errno != 0 || end == capacity_raw || *end != '\0' || parsed < 1 || parsed > 64) {
            result.error = "OMNI_HIFT_PLAN_CACHE_CAPACITY must be an integer in [1, 64]";
            return result;
        }
        result.plan_cache_capacity = static_cast<size_t>(parsed);
    }

    return result;
}

inline hift_runner_config hift_runner_config_from_environment() {
    return parse_hift_runner_config(
        std::getenv("OMNI_HIFT_RUNNER"),
        std::getenv("OMNI_HIFT_PLAN_CACHE_CAPACITY"));
}

}  // namespace omni::flow
