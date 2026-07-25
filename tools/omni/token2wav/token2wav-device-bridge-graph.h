#pragma once

#include "token2wav-device-bridge-policy.h"

#include "ggml.h"

#include <cstdint>
#include <string>

namespace omni::flow {

// Wrap a synchronized external device allocation as a leaf in a new GGML
// graph.  Deliberately do not use ggml_view_* on producer_tensor: a normal
// view retains producer_tensor in src[0], causing bridge graph traversal to
// pull in the complete producer graph.
inline ggml_tensor * make_borrowed_device_leaf(
        ggml_context *                          ctx,
        const ggml_tensor *                     producer_tensor,
        const borrowed_device_tensor_contract & contract,
        const int64_t                           stride_bytes[3],
        std::string &                           error) {
    error.clear();
    if (!ctx || !producer_tensor || !producer_tensor->buffer ||
        !producer_tensor->data ||
        producer_tensor->type != GGML_TYPE_F32 ||
        producer_tensor->data !=
            reinterpret_cast<void *>(contract.data)) {
        error = "borrowed mel producer tensor is not live F32 device storage";
        return nullptr;
    }
    if (!stride_bytes ||
        stride_bytes[0] != (int64_t) sizeof(float) ||
        stride_bytes[1] !=
            contract.channels * (int64_t) sizeof(float) ||
        stride_bytes[2] <
            contract.frames * stride_bytes[1]) {
        error = "borrowed mel tensor is not a contiguous CTB prefix";
        return nullptr;
    }

    ggml_tensor * leaf =
        ggml_new_tensor_3d(
            ctx,
            GGML_TYPE_F32,
            contract.channels,
            contract.frames,
            contract.batch);
    leaf->buffer = producer_tensor->buffer;
    leaf->data = reinterpret_cast<void *>(contract.data);
    ggml_set_input(leaf);
    if (leaf->op != GGML_OP_NONE || leaf->src[0] != nullptr) {
        error = "failed to detach borrowed mel producer lineage";
        return nullptr;
    }
    return leaf;
}

}  // namespace omni::flow
