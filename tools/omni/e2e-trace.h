#pragma once

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#ifndef OMNI_E2E_TRACE_HAS_MSTX
#if __has_include(<mstx/ms_tools_ext.h>)
#define OMNI_E2E_TRACE_HAS_MSTX 1
#else
#define OMNI_E2E_TRACE_HAS_MSTX 0
#endif
#endif

#if OMNI_E2E_TRACE_HAS_MSTX
#include <mstx/ms_tools_ext.h>
#endif

namespace omni::e2e_trace {

constexpr const char * k_schema_version = "minicpmo45-omni-trace-event/v1";

enum class trace_mode {
    off,
    jsonl,
    mstx,
};

enum class tensor_mode {
    off,
    summary,
    jsonl,
};

struct context {
    std::string session_id;
    int64_t     epoch = 0;
    int64_t     turn_id = 0;
    int64_t     frame_id = 0;
    int64_t     chunk_id = 0;
    int64_t     queue_depth = 0;
};

struct tensor_record {
    context     ids;
    std::string stage;
    std::string action;
    std::string tensor_role;
    std::string producer;
    std::string consumer;
    std::string shape;
    std::string dtype;
    std::string device;
    std::string stream;
    std::string transfer;
    std::string residency_claim;
    uint64_t    bytes = 0;
    uintptr_t   buffer_id = 0;
    bool        reused = false;
};

inline context & current_context() {
    static thread_local context value;
    return value;
}

class context_scope {
  public:
    explicit context_scope(context next)
        : previous_(current_context()) {
        current_context() = std::move(next);
    }

    ~context_scope() {
        current_context() = previous_;
    }

    context_scope(const context_scope &) = delete;
    context_scope & operator=(const context_scope &) = delete;

  private:
    context previous_;
};

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(ch)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

inline std::string pointer_id(uintptr_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << value;
    return out.str();
}

class state {
  public:
    bool initialize(
            const char * trace_mode_raw,
            const char * trace_output_raw,
            const char * tensor_mode_raw,
            const char * tensor_output_raw,
            std::string & error) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            error = "trace state already initialized";
            return false;
        }

        if (!parse_trace_mode(trace_mode_raw, trace_mode_, error) ||
            !parse_tensor_mode(tensor_mode_raw, tensor_mode_, error)) {
            return false;
        }

        trace_output_ = trace_output_raw ? trace_output_raw : "";
        tensor_output_ = tensor_output_raw ? tensor_output_raw : "";

        if (trace_mode_ != trace_mode::off && trace_output_.empty()) {
            error = "OMNI_TRACE_OUTPUT is required when OMNI_TRACE is enabled";
            return false;
        }
        if (tensor_mode_ != tensor_mode::off && tensor_output_.empty()) {
            error =
                "OMNI_TENSOR_RESIDENCY_OUTPUT is required when "
                "OMNI_TENSOR_RESIDENCY_TRACE is enabled";
            return false;
        }
#if !OMNI_E2E_TRACE_HAS_MSTX
        if (trace_mode_ == trace_mode::mstx) {
            error = "OMNI_TRACE=mstx requires CANN MSTX headers at build time";
            return false;
        }
#endif

        if (!trace_output_.empty()) {
            std::ofstream out(trace_output_, std::ios::out | std::ios::trunc);
            if (!out) {
                error = "cannot create OMNI_TRACE_OUTPUT: " + trace_output_;
                return false;
            }
        }
        if (!tensor_output_.empty()) {
            std::ofstream out(tensor_output_, std::ios::out | std::ios::trunc);
            if (!out) {
                error =
                    "cannot create OMNI_TENSOR_RESIDENCY_OUTPUT: " +
                    tensor_output_;
                return false;
            }
        }

        initialized_ = true;
        return true;
    }

    bool initialize_from_environment(std::string & error) {
        return initialize(
            std::getenv("OMNI_TRACE"),
            std::getenv("OMNI_TRACE_OUTPUT"),
            std::getenv("OMNI_TENSOR_RESIDENCY_TRACE"),
            std::getenv("OMNI_TENSOR_RESIDENCY_OUTPUT"),
            error);
    }

    bool enabled() const {
        return trace_mode_ != trace_mode::off;
    }

    bool tensor_enabled() const {
        return tensor_mode_ != tensor_mode::off;
    }

    trace_mode mode() const {
        return trace_mode_;
    }

    bool append_span(
            const context & ids,
            const std::string & stage,
            const std::string & action,
            const std::string & device,
            const std::string & stream,
            int64_t start_ns,
            int64_t end_ns) {
        if (!enabled()) {
            return true;
        }
        std::ostringstream row;
        row << "{"
            << "\"schema_version\":\"" << k_schema_version << "\","
            << "\"name\":\"" << json_escape(stage + "." + action) << "\","
            << "\"kind\":\"span\","
            << "\"clock\":\"steady_clock\","
            << "\"start_ns\":" << start_ns << ","
            << "\"end_ns\":" << end_ns << ","
            << "\"stage\":\"" << json_escape(stage) << "\","
            << "\"action\":\"" << json_escape(action) << "\","
            << "\"device\":\"" << json_escape(device) << "\","
            << "\"stream\":\"" << json_escape(stream) << "\","
            << context_json(ids)
            << "}";
        return append_line(trace_output_, row.str());
    }

    bool append_instant(
            const context & ids,
            const std::string & stage,
            const std::string & action,
            const std::string & device = "",
            const std::string & stream = "") {
        if (!enabled()) {
            return true;
        }
        const int64_t timestamp = now_ns();
        std::ostringstream row;
        row << "{"
            << "\"schema_version\":\"" << k_schema_version << "\","
            << "\"name\":\"" << json_escape(stage + "." + action) << "\","
            << "\"kind\":\"instant\","
            << "\"clock\":\"steady_clock\","
            << "\"start_ns\":" << timestamp << ","
            << "\"end_ns\":null,"
            << "\"stage\":\"" << json_escape(stage) << "\","
            << "\"action\":\"" << json_escape(action) << "\","
            << "\"device\":\"" << json_escape(device) << "\","
            << "\"stream\":\"" << json_escape(stream) << "\","
            << context_json(ids)
            << "}";
        return append_line(trace_output_, row.str());
    }

    bool append_tensor(const tensor_record & record) {
        if (!tensor_enabled()) {
            return true;
        }
        std::ostringstream row;
        row << "{"
            << "\"schema_version\":\"minicpmo45-tensor-residency/v1\","
            << "\"kind\":\"tensor\","
            << "\"clock\":\"steady_clock\","
            << "\"timestamp_ns\":" << now_ns() << ","
            << "\"stage\":\"" << json_escape(record.stage) << "\","
            << "\"action\":\"" << json_escape(record.action) << "\","
            << "\"tensor_role\":\"" << json_escape(record.tensor_role) << "\","
            << "\"producer\":\"" << json_escape(record.producer) << "\","
            << "\"consumer\":\"" << json_escape(record.consumer) << "\","
            << "\"shape\":\"" << json_escape(record.shape) << "\","
            << "\"dtype\":\"" << json_escape(record.dtype) << "\","
            << "\"device\":\"" << json_escape(record.device) << "\","
            << "\"stream\":\"" << json_escape(record.stream) << "\","
            << "\"transfer\":\"" << json_escape(record.transfer) << "\","
            << "\"residency_claim\":\""
            << json_escape(record.residency_claim) << "\","
            << "\"bytes\":" << record.bytes << ","
            << "\"buffer_id\":\"" << pointer_id(record.buffer_id) << "\","
            << "\"reused\":" << (record.reused ? "true" : "false") << ","
            << context_json(record.ids)
            << "}";
        return append_line(tensor_output_, row.str());
    }

  private:
    static bool parse_trace_mode(
            const char * raw,
            trace_mode & result,
            std::string & error) {
        const std::string value = raw ? raw : "off";
        if (value == "off") {
            result = trace_mode::off;
        } else if (value == "jsonl") {
            result = trace_mode::jsonl;
        } else if (value == "mstx") {
            result = trace_mode::mstx;
        } else {
            error = "OMNI_TRACE must be off|jsonl|mstx";
            return false;
        }
        return true;
    }

    static bool parse_tensor_mode(
            const char * raw,
            tensor_mode & result,
            std::string & error) {
        const std::string value = raw ? raw : "off";
        if (value == "off") {
            result = tensor_mode::off;
        } else if (value == "summary") {
            result = tensor_mode::summary;
        } else if (value == "jsonl") {
            result = tensor_mode::jsonl;
        } else {
            error =
                "OMNI_TENSOR_RESIDENCY_TRACE must be off|summary|jsonl";
            return false;
        }
        return true;
    }

    static std::string context_json(const context & ids) {
        std::ostringstream out;
        out << "\"session_id\":\"" << json_escape(ids.session_id) << "\","
            << "\"epoch\":" << ids.epoch << ","
            << "\"turn_id\":" << ids.turn_id << ","
            << "\"frame_id\":" << ids.frame_id << ","
            << "\"chunk_id\":" << ids.chunk_id << ","
            << "\"queue_depth\":" << ids.queue_depth << ","
            << "\"thread_id\":\""
            << std::hash<std::thread::id>{}(std::this_thread::get_id())
            << "\"";
        return out.str();
    }

    bool append_line(const std::string & path, const std::string & line) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) {
            return false;
        }
        out << line << '\n';
        return static_cast<bool>(out);
    }

    bool        initialized_ = false;
    trace_mode  trace_mode_ = trace_mode::off;
    tensor_mode tensor_mode_ = tensor_mode::off;
    std::string trace_output_;
    std::string tensor_output_;
    mutable std::mutex mutex_;
};

inline state & global_state() {
    static state singleton;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string error;
        if (!singleton.initialize_from_environment(error)) {
            std::fprintf(stderr, "OMNI E2E trace configuration error: %s\n",
                         error.c_str());
            std::abort();
        }
    });
    return singleton;
}

class span {
  public:
    span(
            std::string stage,
            std::string action,
            std::string device = "",
            std::string stream = "")
        : span(
              current_context(),
              std::move(stage),
              std::move(action),
              std::move(device),
              std::move(stream)) {
    }

    span(
            const context & ids,
            std::string stage,
            std::string action,
            std::string device = "",
            std::string stream = "")
        : trace_(&global_state()),
          ids_(ids),
          stage_(std::move(stage)),
          action_(std::move(action)),
          device_(std::move(device)),
          stream_(std::move(stream)),
          start_ns_(now_ns()) {
#if OMNI_E2E_TRACE_HAS_MSTX
        if (trace_->mode() == trace_mode::mstx) {
            const std::string label = stage_ + "." + action_;
            mstx_id_ = mstxRangeStartA(label.c_str(), nullptr);
        }
#endif
    }

    ~span() {
        const int64_t end_ns = now_ns();
#if OMNI_E2E_TRACE_HAS_MSTX
        if (trace_->mode() == trace_mode::mstx &&
            mstx_id_ != MSTX_TOOL_INVALID_ID) {
            mstxRangeEnd(mstx_id_);
        }
#endif
        trace_->append_span(
            ids_, stage_, action_, device_, stream_, start_ns_, end_ns);
    }

    span(const span &) = delete;
    span & operator=(const span &) = delete;

  private:
    state *     trace_;
    context     ids_;
    std::string stage_;
    std::string action_;
    std::string device_;
    std::string stream_;
    int64_t     start_ns_;
#if OMNI_E2E_TRACE_HAS_MSTX
    mstxRangeId mstx_id_ = MSTX_TOOL_INVALID_ID;
#endif
};

}  // namespace omni::e2e_trace
