#include "atb_ops.h"

#include "common.h"

#include <atb/atb_infer.h>

#include <algorithm>
#include <vector>

struct ggml_cann_atb_runtime {
    atb::Context * context = nullptr;
    atb::Operation * reshape_and_cache = nullptr;
    void * workspace = nullptr;
    uint64_t workspace_size = 0;
};

static void ggml_cann_atb_check(atb::Status status, const char * operation) {
    if (status != atb::NO_ERROR) {
        GGML_ABORT("ATB %s failed with status %d", operation, static_cast<int>(status));
    }
}

static atb::Tensor ggml_cann_atb_tensor(
        aclDataType dtype,
        const std::vector<int64_t> & dims,
        void * data,
        uint64_t bytes) {
    atb::Tensor tensor;
    tensor.desc.dtype = dtype;
    tensor.desc.format = ACL_FORMAT_ND;
    tensor.desc.shape.dimNum = dims.size();
    std::copy(dims.begin(), dims.end(), tensor.desc.shape.dims);
    tensor.deviceData = data;
    tensor.dataSize = bytes;
    return tensor;
}

ggml_cann_atb_runtime * ggml_cann_atb_runtime_create(aclrtStream stream) {
    auto * runtime = new ggml_cann_atb_runtime;
    ggml_cann_atb_check(atb::CreateContext(&runtime->context), "CreateContext");
    ggml_cann_atb_check(runtime->context->SetExecuteStream(stream), "SetExecuteStream");
    atb::infer::ReshapeAndCacheParam param;
    ggml_cann_atb_check(
        atb::CreateOperation(param, &runtime->reshape_and_cache),
        "CreateOperation(ReshapeAndCache)");
    return runtime;
}

void ggml_cann_atb_runtime_destroy(ggml_cann_atb_runtime * runtime) {
    if (runtime == nullptr) {
        return;
    }
    if (runtime->workspace != nullptr) {
        ACL_CHECK(aclrtFree(runtime->workspace));
    }
    if (runtime->reshape_and_cache != nullptr) {
        ggml_cann_atb_check(atb::DestroyOperation(runtime->reshape_and_cache), "DestroyOperation");
    }
    if (runtime->context != nullptr) {
        ggml_cann_atb_check(atb::DestroyContext(runtime->context), "DestroyContext");
    }
    delete runtime;
}

void ggml_cann_atb_reshape_and_cache(
        ggml_cann_atb_runtime * runtime,
        void * key,
        void * value,
        void * key_cache,
        void * value_cache,
        void * slots_i32,
        int64_t token_count,
        int64_t head_width,
        int64_t total_slots) {
    GGML_ASSERT(runtime != nullptr);
    const uint64_t token_bytes =
        static_cast<uint64_t>(token_count * head_width * sizeof(uint16_t));
    const uint64_t cache_bytes =
        static_cast<uint64_t>(total_slots * head_width * sizeof(uint16_t));
    const uint64_t slot_bytes =
        static_cast<uint64_t>(token_count * sizeof(int32_t));

    atb::VariantPack pack;
    pack.inTensors = {
        ggml_cann_atb_tensor(ACL_FLOAT16, {token_count, 1, head_width}, key, token_bytes),
        ggml_cann_atb_tensor(ACL_FLOAT16, {token_count, 1, head_width}, value, token_bytes),
        ggml_cann_atb_tensor(ACL_FLOAT16, {1, total_slots, 1, head_width}, key_cache, cache_bytes),
        ggml_cann_atb_tensor(ACL_FLOAT16, {1, total_slots, 1, head_width}, value_cache, cache_bytes),
        ggml_cann_atb_tensor(ACL_INT32, {token_count}, slots_i32, slot_bytes),
    };
    pack.outTensors = {
        ggml_cann_atb_tensor(ACL_FLOAT16, {1, total_slots, 1, head_width}, key_cache, cache_bytes),
        ggml_cann_atb_tensor(ACL_FLOAT16, {1, total_slots, 1, head_width}, value_cache, cache_bytes),
    };

    uint64_t requested_workspace = 0;
    ggml_cann_atb_check(
        runtime->reshape_and_cache->Setup(pack, requested_workspace, runtime->context),
        "ReshapeAndCache::Setup");
    if (requested_workspace > runtime->workspace_size) {
        if (runtime->workspace != nullptr) {
            ACL_CHECK(aclrtFree(runtime->workspace));
        }
        runtime->workspace = nullptr;
        runtime->workspace_size = 0;
        if (requested_workspace > 0) {
            ACL_CHECK(aclrtMalloc(
                &runtime->workspace,
                requested_workspace,
                ACL_MEM_MALLOC_HUGE_FIRST));
            runtime->workspace_size = requested_workspace;
        }
    }
    ggml_cann_atb_check(
        runtime->reshape_and_cache->Execute(
            pack,
            static_cast<uint8_t *>(runtime->workspace),
            requested_workspace,
            runtime->context),
        "ReshapeAndCache::Execute");
}
