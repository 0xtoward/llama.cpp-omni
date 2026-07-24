#pragma once

#include <acl/acl.h>
#include <cstdint>

struct ggml_cann_atb_runtime;

#ifdef GGML_CANN_USE_ATB
ggml_cann_atb_runtime * ggml_cann_atb_runtime_create(aclrtStream stream);
void ggml_cann_atb_runtime_destroy(ggml_cann_atb_runtime * runtime);

void ggml_cann_atb_reshape_and_cache(
    ggml_cann_atb_runtime * runtime,
    void * key,
    void * value,
    void * key_cache,
    void * value_cache,
    void * slots_i32,
    int64_t token_count,
    int64_t head_width,
    int64_t total_slots);
#endif
