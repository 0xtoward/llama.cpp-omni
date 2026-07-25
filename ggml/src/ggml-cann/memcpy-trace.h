#pragma once

#include "ggml.h"

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>

enum class ggml_cann_memcpy_direction {
    h2d,
    d2h,
    d2d,
};

enum class ggml_cann_memcpy_mode {
    sync,
    async,
};

// Record one successful CANN memcpy call. The host duration only covers the
// runtime API invocation. Device task time is joined later from msprof using
// the timestamp/thread/API fields; it must never be inferred from this value.
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
    long long end_ns);

long long ggml_cann_memcpy_trace_now_ns();

aclError ggml_cann_memcpy_sync_traced(
    void * dst,
    size_t dest_max,
    const void * src,
    size_t count,
    aclrtMemcpyKind kind,
    const ggml_tensor * tensor,
    const char * phase,
    int device);

aclError ggml_cann_memcpy_async_traced(
    void * dst,
    size_t dest_max,
    const void * src,
    size_t count,
    aclrtMemcpyKind kind,
    aclrtStream stream,
    const ggml_tensor * tensor,
    const char * phase,
    int device);
