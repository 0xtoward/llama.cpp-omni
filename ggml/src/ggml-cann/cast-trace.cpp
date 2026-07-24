#include "cast-trace.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace ggml_cann_cast_trace {
namespace {

struct aggregate {
    event    descriptor;
    uint64_t calls = 0;
    uint64_t bytes = 0;
    uint64_t host_submit_ns_total = 0;
};

struct state {
    std::mutex                       mutex;
    std::map<std::string, aggregate> summaries;
    bool                             summary_flushed = false;
};

state & global_state() {
    static state * value = new state();
    return *value;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    static const char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

std::string summary_key(const event & value) {
    std::ostringstream out;
    out << domain_name(value.cast_domain) << '\x1f'
        << origin_name(value.cast_origin) << '\x1f'
        << value.producer << '\x1f'
        << value.consumer << '\x1f'
        << value.src_dtype << '\x1f'
        << value.dst_dtype;
    for (const int64_t dim : value.shape) {
        out << '\x1f' << dim;
    }
    return out.str();
}

void flush_summary_at_exit() {
    flush_summary();
}

}  // namespace

mode parse_mode(const std::string & value) {
    if (value.empty() || value == "off") {
        return mode::off;
    }
    if (value == "summary") {
        return mode::summary;
    }
    if (value == "jsonl") {
        return mode::jsonl;
    }
    throw std::invalid_argument(
        "GGML_CANN_CAST_TRACE must be one of: off, summary, jsonl");
}

const char * mode_name(mode value) {
    switch (value) {
        case mode::off:
            return "off";
        case mode::summary:
            return "summary";
        case mode::jsonl:
            return "jsonl";
    }
    return "invalid";
}

const char * domain_name(domain value) {
    switch (value) {
        case domain::generic:
            return "generic";
        case domain::thinker:
            return "thinker";
        case domain::tts:
            return "tts";
        case domain::token2mel:
            return "token2mel";
        case domain::hift:
            return "hift";
    }
    return "invalid";
}

const char * origin_name(origin value) {
    switch (value) {
        case origin::generic:
            return "generic";
        case origin::kv:
            return "kv";
        case origin::fia:
            return "fia";
        case origin::q8:
            return "q8";
    }
    return "invalid";
}

domain infer_domain(const std::string & producer, const std::string & consumer) {
    const std::string names = lowercase(producer + " " + consumer);
    if (names.find("token2mel") != std::string::npos ||
        names.find("token_to_mel") != std::string::npos) {
        return domain::token2mel;
    }
    if (names.find("hift") != std::string::npos) {
        return domain::hift;
    }
    if (names.find("minicpmtts") != std::string::npos ||
        names.find("talker") != std::string::npos ||
        names.find("tts") != std::string::npos) {
        return domain::tts;
    }
    if (names.find("thinker") != std::string::npos ||
        names.find("qwen") != std::string::npos ||
        names.find("cache_k_") != std::string::npos ||
        names.find("cache_v_") != std::string::npos ||
        names.find("fattn") != std::string::npos) {
        return domain::thinker;
    }
    return domain::generic;
}

mode configured_mode() {
    static const mode value = [] {
        const char * raw = std::getenv("GGML_CANN_CAST_TRACE");
        try {
            const mode parsed = parse_mode(raw ? raw : "");
            if (parsed == mode::summary) {
                std::atexit(flush_summary_at_exit);
            }
            return parsed;
        } catch (const std::invalid_argument & error) {
            std::fprintf(stderr,
                         "CANN_CAST_TRACE_CONFIG_ERROR %s; got=%s; "
                         "falling_back=off\n",
                         error.what(), raw ? raw : "");
            std::fflush(stderr);
            return mode::off;
        }
    }();
    return value;
}

bool enabled() {
    return configured_mode() != mode::off;
}

void report_error(const char * stage) noexcept {
    static std::atomic_flag reported = ATOMIC_FLAG_INIT;
    if (!reported.test_and_set()) {
        std::fprintf(stderr,
                     "CANN_CAST_TRACE_ERROR stage=%s "
                     "action=disable_current_event\n",
                     stage ? stage : "unknown");
        std::fflush(stderr);
    }
}

std::string event_json(const event & value, uint64_t calls, uint64_t bytes,
                       uint64_t host_submit_ns_total) {
    std::ostringstream out;
    out << '{'
        << "\"domain\":\"" << domain_name(value.cast_domain) << "\","
        << "\"origin\":\"" << origin_name(value.cast_origin) << "\","
        << "\"producer\":\"" << json_escape(value.producer) << "\","
        << "\"consumer\":\"" << json_escape(value.consumer) << "\","
        << "\"src_dtype\":\"" << json_escape(value.src_dtype) << "\","
        << "\"dst_dtype\":\"" << json_escape(value.dst_dtype) << "\","
        << "\"shape\":[";
    for (size_t i = 0; i < value.shape.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << value.shape[i];
    }
    out << "],"
        << "\"calls\":" << calls << ','
        << "\"bytes\":" << bytes << ','
        << "\"host_submit_ns_total\":" << host_submit_ns_total << ','
        << "\"host_submit_ns_avg\":"
        << (calls == 0 ? 0 : host_submit_ns_total / calls) << ','
        << "\"host_api_time_ns_total\":" << host_submit_ns_total << ','
        << "\"host_api_time_ns_avg\":"
        << (calls == 0 ? 0 : host_submit_ns_total / calls) << ','
        << "\"host_api_time_source\":"
           "\"aclnn_workspace_and_async_launch_wall\","
        << "\"device_time_ns\":null,"
        << "\"device_time_source\":\"msprof_postprocess\""
        << '}';
    return out.str();
}

void record(const event & value, uint64_t host_submit_ns) noexcept {
    try {
        const mode trace_mode = configured_mode();
        if (trace_mode == mode::off) {
            return;
        }

        if (trace_mode == mode::jsonl) {
            const std::string json =
                event_json(value, 1, value.bytes, host_submit_ns);
            std::fprintf(stderr, "CANN_CAST_TRACE_EVENT %s\n", json.c_str());
            std::fflush(stderr);
            return;
        }

        state & trace_state = global_state();
        std::lock_guard<std::mutex> lock(trace_state.mutex);
        aggregate & total = trace_state.summaries[summary_key(value)];
        if (total.calls == 0) {
            total.descriptor = value;
        }
        total.calls++;
        total.bytes += value.bytes;
        total.host_submit_ns_total += host_submit_ns;
    } catch (...) {
        report_error("record");
    }
}

void flush_summary() noexcept {
    try {
        if (configured_mode() != mode::summary) {
            return;
        }

        state & trace_state = global_state();
        std::lock_guard<std::mutex> lock(trace_state.mutex);
        if (trace_state.summary_flushed) {
            return;
        }
        for (const auto & item : trace_state.summaries) {
            const aggregate & total = item.second;
            const std::string json =
                event_json(total.descriptor, total.calls, total.bytes,
                           total.host_submit_ns_total);
            std::fprintf(stderr, "CANN_CAST_TRACE_SUMMARY %s\n", json.c_str());
        }
        std::fflush(stderr);
        trace_state.summary_flushed = true;
    } catch (...) {
        report_error("flush");
    }
}

}  // namespace ggml_cann_cast_trace
