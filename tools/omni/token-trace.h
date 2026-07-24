#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace omni::token_trace {

constexpr size_t k_teacher_token_count = 32;

struct top_k_entry {
    int32_t token = -1;
    float logit = 0.0f;
};

struct hidden_fingerprint {
    size_t count = 0;
    size_t finite_count = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    float min = 0.0f;
    float max = 0.0f;
    double mean = 0.0;
    double l2 = 0.0;
    uint64_t hash = 1469598103934665603ULL;
};

struct state {
    std::string trace_path;
    std::vector<int32_t> teacher_tokens;
    size_t teacher_index = 0;
    uint64_t step = 0;
    uint64_t session_generation = 0;
    int prompt_token_count = 0;
    int request_prefill_token_count = 0;

    bool enabled() const {
        return !trace_path.empty();
    }

    bool teacher_enabled() const {
        return !teacher_tokens.empty();
    }
};

struct record {
    uint64_t step = 0;
    int position = 0;
    int n_past_before = 0;
    int n_past_after = 0;
    int turn_id = 0;
    uint64_t session_generation = 0;
    int prompt_token_count = 0;
    int request_prefill_token_count = 0;
    std::string session_hint;
    int32_t greedy_token = -1;
    int32_t sampled_token = -1;
    int32_t selected_token = -1;
    std::string token_type;
    bool teacher_active = false;
    size_t teacher_index = 0;
    bool raw_logits_available = false;
    std::vector<top_k_entry> top_k;
    bool hidden_available = false;
    hidden_fingerprint hidden;
};

inline bool parse_teacher_tokens_file(
        const std::string & path,
        std::vector<int32_t> & tokens,
        std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open teacher token file: " + path;
        return false;
    }

    tokens.clear();
    std::string item;
    while (input >> item) {
        if (item.empty() ||
            !std::all_of(item.begin(), item.end(), [](unsigned char ch) {
                return ch >= '0' && ch <= '9';
            })) {
            error = "teacher tokens must be unsigned decimal integers";
            return false;
        }

        uint64_t value = 0;
        for (char ch : item) {
            const uint64_t digit = static_cast<uint64_t>(ch - '0');
            if (value > (static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) - digit) / 10) {
                error = "teacher token exceeds int32 range";
                return false;
            }
            value = value * 10 + digit;
        }
        tokens.push_back(static_cast<int32_t>(value));
        if (tokens.size() > k_teacher_token_count) {
            error = "teacher token file must contain exactly 32 tokens";
            return false;
        }
    }

    if (!input.eof()) {
        error = "failed while reading teacher token file";
        return false;
    }
    if (tokens.size() != k_teacher_token_count) {
        error = "teacher token file must contain exactly 32 tokens";
        return false;
    }
    return true;
}

inline bool initialize(
        const char * trace_path_env,
        const char * teacher_path_env,
        state & result,
        std::string & error) {
    result = {};
    const std::string trace_path = trace_path_env ? trace_path_env : "";
    const std::string teacher_path = teacher_path_env ? teacher_path_env : "";

    if (trace_path.empty() && teacher_path.empty()) {
        return true;
    }
    if (trace_path.empty()) {
        error = "OMNI_TEACHER_TOKENS_FILE requires OMNI_TOKEN_TRACE";
        return false;
    }

    {
        std::ofstream trace(trace_path, std::ios::out | std::ios::trunc);
        if (!trace) {
            error = "cannot create OMNI_TOKEN_TRACE file: " + trace_path;
            return false;
        }
    }

    result.trace_path = trace_path;
    if (!teacher_path.empty() &&
        !parse_teacher_tokens_file(teacher_path, result.teacher_tokens, error)) {
        result = {};
        return false;
    }
    return true;
}

inline bool next_teacher_token(
        state & debug,
        int32_t & token,
        size_t & index,
        std::string & error) {
    if (!debug.teacher_enabled()) {
        error = "teacher mode is not enabled";
        return false;
    }
    if (debug.teacher_index >= debug.teacher_tokens.size()) {
        error = "teacher token sequence exhausted after 32 decode steps";
        return false;
    }
    index = debug.teacher_index;
    token = debug.teacher_tokens[debug.teacher_index++];
    return true;
}

inline bool teacher_sequence_complete(const state & debug) {
    return debug.teacher_enabled() &&
           debug.teacher_index == debug.teacher_tokens.size();
}

inline std::vector<top_k_entry> compute_top_k(
        const float * logits,
        size_t count,
        size_t k) {
    std::vector<top_k_entry> entries;
    if (logits == nullptr || count == 0 || k == 0) {
        return entries;
    }
    entries.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (!std::isnan(logits[i])) {
            entries.push_back({static_cast<int32_t>(i), logits[i]});
        }
    }
    const auto better = [](const top_k_entry & lhs, const top_k_entry & rhs) {
        if (lhs.logit != rhs.logit) {
            return lhs.logit > rhs.logit;
        }
        return lhs.token < rhs.token;
    };
    const size_t keep = std::min(k, entries.size());
    std::partial_sort(entries.begin(), entries.begin() + keep, entries.end(), better);
    entries.resize(keep);
    return entries;
}

inline hidden_fingerprint fingerprint(const float * values, size_t count) {
    hidden_fingerprint result;
    result.count = count;
    if (values == nullptr || count == 0) {
        return result;
    }

    double sum = 0.0;
    double square_sum = 0.0;
    bool have_finite = false;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(values[i]));
        std::memcpy(&bits, values + i, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            result.hash ^= static_cast<uint8_t>((bits >> (byte * 8)) & 0xffU);
            result.hash *= 1099511628211ULL;
        }

        const float value = values[i];
        if (std::isnan(value)) {
            ++result.nan_count;
            continue;
        }
        if (std::isinf(value)) {
            ++result.inf_count;
            continue;
        }
        if (!have_finite) {
            result.min = value;
            result.max = value;
            have_finite = true;
        } else {
            result.min = std::min(result.min, value);
            result.max = std::max(result.max, value);
        }
        ++result.finite_count;
        sum += value;
        square_sum += static_cast<double>(value) * value;
    }
    if (result.finite_count > 0) {
        result.mean = sum / static_cast<double>(result.finite_count);
        result.l2 = std::sqrt(square_sum);
    }
    return result;
}

inline std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

inline void append_json_number(std::ostringstream & out, double value) {
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
}

inline std::string to_json(const record & event) {
    std::ostringstream out;
    out << std::setprecision(9);
    out << "{"
        << "\"event\":\"omni_token_trace\""
        << ",\"step\":" << event.step
        << ",\"position\":" << event.position
        << ",\"n_past_before\":" << event.n_past_before
        << ",\"n_past_after\":" << event.n_past_after
        << ",\"turn_id\":" << event.turn_id
        << ",\"session_generation\":" << event.session_generation
        << ",\"prompt_token_count\":" << event.prompt_token_count
        << ",\"request_prefill_token_count\":"
        << event.request_prefill_token_count
        << ",\"session_hint\":\"" << json_escape(event.session_hint) << "\""
        << ",\"greedy_token\":" << event.greedy_token
        << ",\"sampled_token\":" << event.sampled_token
        << ",\"selected_token\":" << event.selected_token
        << ",\"token_type\":\"" << json_escape(event.token_type) << "\""
        << ",\"teacher_active\":" << (event.teacher_active ? "true" : "false")
        << ",\"teacher_index\":";
    if (event.teacher_active) {
        out << event.teacher_index;
    } else {
        out << "null";
    }
    out << ",\"raw_logits_available\":"
        << (event.raw_logits_available ? "true" : "false")
        << ",\"top_k\":";
    if (!event.raw_logits_available) {
        out << "null";
    } else {
        out << "[";
        for (size_t i = 0; i < event.top_k.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << "{\"token\":" << event.top_k[i].token
                << ",\"logit\":";
            append_json_number(out, event.top_k[i].logit);
            out << "}";
        }
        out << "]";
    }
    out << ",\"hidden\":";
    if (!event.hidden_available) {
        out << "null";
    } else {
        out << "{"
            << "\"count\":" << event.hidden.count
            << ",\"finite\":" << event.hidden.finite_count
            << ",\"nan\":" << event.hidden.nan_count
            << ",\"inf\":" << event.hidden.inf_count
            << ",\"min\":";
        append_json_number(out, event.hidden.min);
        out << ",\"max\":";
        append_json_number(out, event.hidden.max);
        out << ",\"mean\":";
        append_json_number(out, event.hidden.mean);
        out << ",\"l2\":";
        append_json_number(out, event.hidden.l2);
        out << ",\"fnv1a64\":\""
            << std::hex << std::setw(16) << std::setfill('0') << event.hidden.hash
            << std::dec << "\"}";
    }
    out << "}";
    return out.str();
}

inline bool append_jsonl(
        const state & debug,
        const record & event,
        std::string & error) {
    if (!debug.enabled()) {
        return true;
    }
    std::ofstream trace(debug.trace_path, std::ios::out | std::ios::app);
    if (!trace) {
        error = "cannot append OMNI_TOKEN_TRACE file: " + debug.trace_path;
        return false;
    }
    trace << to_json(event) << "\n";
    trace.flush();
    if (!trace) {
        error = "failed writing OMNI_TOKEN_TRACE file: " + debug.trace_path;
        return false;
    }
    return true;
}

}  // namespace omni::token_trace
