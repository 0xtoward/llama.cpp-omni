#ifndef GGML_CANN_CAST_TRACE_H
#define GGML_CANN_CAST_TRACE_H

#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cann_cast_trace {

enum class mode {
    off,
    summary,
    jsonl,
};

enum class domain {
    generic,
    thinker,
    tts,
    token2mel,
    hift,
};

enum class origin {
    generic,
    kv,
    fia,
    q8,
};

struct event {
    domain               cast_domain = domain::generic;
    origin               cast_origin = origin::generic;
    std::string          producer;
    std::string          consumer;
    std::string          src_dtype;
    std::string          dst_dtype;
    std::vector<int64_t> shape;
    uint64_t             bytes = 0;
};

mode parse_mode(const std::string & value);
const char * mode_name(mode value);
const char * domain_name(domain value);
const char * origin_name(origin value);

// Infer only when a call site has useful tensor/operator names. Unknown names
// deliberately remain generic instead of guessing a stage.
domain infer_domain(const std::string & producer, const std::string & consumer);

// The process-wide mode is parsed once from GGML_CANN_CAST_TRACE. Invalid
// values emit one configuration error and fall back to off so instrumentation
// can never prevent inference. The default is off.
mode configured_mode();
bool enabled();

// host_submit_ns measures only host-side aclnn workspace/executor preparation
// and asynchronous launch submission. It is not device execution time.
void record(const event & value, uint64_t host_submit_ns) noexcept;
void flush_summary() noexcept;
void report_error(const char * stage) noexcept;

// Exposed for the independent CPU-only contract test.
std::string event_json(const event & value, uint64_t calls, uint64_t bytes,
                       uint64_t host_submit_ns_total);

}  // namespace ggml_cann_cast_trace

#endif  // GGML_CANN_CAST_TRACE_H
