#include "memcpy-trace.h"

#include "ggml-impl.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

namespace {

enum class trace_mode {
    off,
    summary,
    jsonl,
};

const char * direction_name(ggml_cann_memcpy_direction direction) {
    switch (direction) {
        case ggml_cann_memcpy_direction::h2d: return "h2d";
        case ggml_cann_memcpy_direction::d2h: return "d2h";
        case ggml_cann_memcpy_direction::d2d: return "d2d";
    }
    return "unknown";
}

const char * mode_name(ggml_cann_memcpy_mode mode) {
    return mode == ggml_cann_memcpy_mode::async ? "async" : "sync";
}

size_t counter_index(ggml_cann_memcpy_direction direction, ggml_cann_memcpy_mode mode) {
    return static_cast<size_t>(direction) * 2 + static_cast<size_t>(mode);
}

std::string json_escape(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    for (const unsigned char ch : input) {
        switch (ch) {
            case '"':  output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char escaped[7];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", ch);
                    output += escaped;
                } else {
                    output += static_cast<char>(ch);
                }
        }
    }
    return output;
}

struct trace_state {
    trace_mode mode = trace_mode::off;
    std::mutex mutex;
    std::ofstream output;
    std::array<std::atomic<uint64_t>, 6> calls {};
    std::array<std::atomic<uint64_t>, 6> bytes {};
    std::array<std::atomic<uint64_t>, 6> host_ns {};

    trace_state() {
        const char * raw = std::getenv("GGML_CANN_MEMCPY_TRACE");
        const std::string value = raw == nullptr ? "off" : raw;
        if (value == "off") {
            mode = trace_mode::off;
        } else if (value == "summary") {
            mode = trace_mode::summary;
        } else if (value == "jsonl") {
            mode = trace_mode::jsonl;
            const char * path = std::getenv("GGML_CANN_MEMCPY_TRACE_OUTPUT");
            if (path == nullptr || path[0] == '\0') {
                GGML_ABORT(
                    "GGML_CANN_MEMCPY_TRACE=jsonl requires "
                    "GGML_CANN_MEMCPY_TRACE_OUTPUT");
            }
            output.open(path, std::ios::out | std::ios::app);
            if (!output.good()) {
                GGML_ABORT("failed to open GGML_CANN_MEMCPY_TRACE_OUTPUT=%s", path);
            }
        } else {
            GGML_ABORT(
                "invalid GGML_CANN_MEMCPY_TRACE=%s; expected off|summary|jsonl",
                value.c_str());
        }
    }

    ~trace_state() {
        if (mode == trace_mode::off) {
            return;
        }
        for (size_t direction = 0; direction < 3; ++direction) {
            for (size_t async = 0; async < 2; ++async) {
                const size_t index = direction * 2 + async;
                GGML_LOG_INFO(
                    "CANN memcpy trace: direction=%s mode=%s calls=%llu "
                    "bytes=%llu host_api_ms=%.3f\n",
                    direction_name(static_cast<ggml_cann_memcpy_direction>(direction)),
                    mode_name(static_cast<ggml_cann_memcpy_mode>(async)),
                    static_cast<unsigned long long>(calls[index].load()),
                    static_cast<unsigned long long>(bytes[index].load()),
                    host_ns[index].load() / 1.0e6);
            }
        }
    }
};

trace_state & state() {
    static trace_state instance;
    return instance;
}

ggml_cann_memcpy_direction direction_from_acl(aclrtMemcpyKind kind) {
    switch (kind) {
        case ACL_MEMCPY_HOST_TO_DEVICE:
            return ggml_cann_memcpy_direction::h2d;
        case ACL_MEMCPY_DEVICE_TO_HOST:
            return ggml_cann_memcpy_direction::d2h;
        case ACL_MEMCPY_DEVICE_TO_DEVICE:
            return ggml_cann_memcpy_direction::d2d;
        default:
            GGML_ABORT("unsupported aclrtMemcpyKind=%d in CANN memcpy trace", static_cast<int>(kind));
    }
}

std::string tensor_shape(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "[]";
    }
    char buffer[160];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "[%lld,%lld,%lld,%lld]",
        static_cast<long long>(tensor->ne[0]),
        static_cast<long long>(tensor->ne[1]),
        static_cast<long long>(tensor->ne[2]),
        static_cast<long long>(tensor->ne[3]));
    return buffer;
}

} // namespace

long long ggml_cann_memcpy_trace_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void ggml_cann_memcpy_trace_record(
    ggml_cann_memcpy_direction direction,
    ggml_cann_memcpy_mode mode,
    size_t bytes,
    const ggml_tensor * tensor,
    const char * backend_api,
    const char * phase,
    int device,
    uintptr_t stream_id,
    long long start_ns,
    long long end_ns) {
    trace_state & trace = state();
    if (trace.mode == trace_mode::off) {
        return;
    }

    const size_t index = counter_index(direction, mode);
    trace.calls[index].fetch_add(1, std::memory_order_relaxed);
    trace.bytes[index].fetch_add(bytes, std::memory_order_relaxed);
    trace.host_ns[index].fetch_add(
        static_cast<uint64_t>(end_ns - start_ns),
        std::memory_order_relaxed);

    if (trace.mode != trace_mode::jsonl) {
        return;
    }

    const char * name = tensor == nullptr || tensor->name[0] == '\0'
                            ? "unnamed"
                            : tensor->name;
    const char * dtype = tensor == nullptr ? "unknown" : ggml_type_name(tensor->type);
    const auto thread_id = std::hash<std::thread::id> {}(std::this_thread::get_id());
    const std::string shape = tensor_shape(tensor);

    std::lock_guard<std::mutex> lock(trace.mutex);
    trace.output
        << "{\"schema\":\"ggml_cann_memcpy_v2\""
        << ",\"direction\":\"" << direction_name(direction) << "\""
        << ",\"mode\":\"" << mode_name(mode) << "\""
        << ",\"bytes\":" << bytes
        << ",\"tensor\":\"" << json_escape(name) << "\""
        << ",\"shape\":" << shape
        << ",\"dtype\":\"" << json_escape(dtype) << "\""
        << ",\"backend_api\":\"" << json_escape(backend_api == nullptr ? "" : backend_api) << "\""
        << ",\"phase\":\"" << json_escape(phase == nullptr ? "unknown" : phase) << "\""
        << ",\"device\":" << device
        << ",\"stream_id\":" << stream_id
        << ",\"thread_id\":" << thread_id
        << ",\"connection_id\":null"
        << ",\"task_id\":null"
        << ",\"start_ns\":" << start_ns
        << ",\"end_ns\":" << end_ns
        << ",\"host_api_ns\":" << (end_ns - start_ns)
        << ",\"device_task_ns\":null"
        << "}\n";
    trace.output.flush();
}

aclError ggml_cann_memcpy_sync_traced(
    void * dst,
    size_t dest_max,
    const void * src,
    size_t count,
    aclrtMemcpyKind kind,
    const ggml_tensor * tensor,
    const char * phase,
    int device) {
    trace_state & trace = state();
    if (trace.mode == trace_mode::off) {
        return aclrtMemcpy(dst, dest_max, src, count, kind);
    }
    const long long start_ns = ggml_cann_memcpy_trace_now_ns();
    const aclError result = aclrtMemcpy(dst, dest_max, src, count, kind);
    const long long end_ns = ggml_cann_memcpy_trace_now_ns();
    if (result == ACL_SUCCESS) {
        ggml_cann_memcpy_trace_record(
            direction_from_acl(kind),
            ggml_cann_memcpy_mode::sync,
            count,
            tensor,
            "aclrtMemcpy",
            phase,
            device,
            0,
            start_ns,
            end_ns);
    }
    return result;
}

aclError ggml_cann_memcpy_async_traced(
    void * dst,
    size_t dest_max,
    const void * src,
    size_t count,
    aclrtMemcpyKind kind,
    aclrtStream stream,
    const ggml_tensor * tensor,
    const char * phase,
    int device) {
    trace_state & trace = state();
    if (trace.mode == trace_mode::off) {
        return aclrtMemcpyAsync(dst, dest_max, src, count, kind, stream);
    }
    const long long start_ns = ggml_cann_memcpy_trace_now_ns();
    const aclError result = aclrtMemcpyAsync(dst, dest_max, src, count, kind, stream);
    const long long end_ns = ggml_cann_memcpy_trace_now_ns();
    if (result == ACL_SUCCESS) {
        ggml_cann_memcpy_trace_record(
            direction_from_acl(kind),
            ggml_cann_memcpy_mode::async,
            count,
            tensor,
            "aclrtMemcpyAsync",
            phase,
            device,
            reinterpret_cast<uintptr_t>(stream),
            start_ns,
            end_ns);
    }
    return result;
}
