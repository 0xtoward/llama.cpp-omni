#include "token2wav/token2wav-device-bridge-graph.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>

int main() {
    constexpr int kProducerNodes = 96;
    ggml_init_params producer_params{};
    producer_params.mem_size =
        256 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(256, false);
    producer_params.no_alloc = true;
    ggml_context * producer_ctx = ggml_init(producer_params);
    if (!producer_ctx) {
        return 1;
    }

    ggml_tensor * input =
        ggml_new_tensor_3d(
            producer_ctx, GGML_TYPE_F32, 80, 50, 1);
    ggml_set_input(input);
    ggml_tensor * producer_output = input;
    for (int i = 0; i < kProducerNodes; ++i) {
        producer_output = ggml_dup(producer_ctx, producer_output);
    }
    ggml_cgraph * producer_graph =
        ggml_new_graph_custom(producer_ctx, 256, false);
    ggml_build_forward_expand(producer_graph, producer_output);
    if (ggml_graph_n_nodes(producer_graph) <= 64) {
        ggml_free(producer_ctx);
        return 2;
    }

    ggml_backend_t backend =
        ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        ggml_free(producer_ctx);
        return 3;
    }
    ggml_backend_buffer_t producer_buffer =
        ggml_backend_alloc_ctx_tensors(producer_ctx, backend);
    if (!producer_buffer || !producer_output->buffer ||
        !producer_output->data) {
        if (producer_buffer) {
            ggml_backend_buffer_free(producer_buffer);
        }
        ggml_backend_free(backend);
        ggml_free(producer_ctx);
        return 4;
    }

    omni::flow::borrowed_device_tensor_contract contract;
    contract.source_device = 1;
    contract.data =
        reinterpret_cast<uintptr_t>(producer_output->data);
    contract.channels = 80;
    contract.frames = 50;
    contract.batch = 1;
    contract.bytes = 80 * 50 * sizeof(float);
    contract.dtype = omni::flow::device_bridge_dtype::f32;
    contract.producer_synchronized = true;
    contract.epoch = 1;
    const int64_t strides[3] = {
        static_cast<int64_t>(producer_output->nb[0]),
        static_cast<int64_t>(producer_output->nb[1]),
        static_cast<int64_t>(producer_output->nb[2]),
    };

    ggml_init_params bridge_params{};
    bridge_params.mem_size =
        32 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(64, false);
    bridge_params.no_alloc = true;
    ggml_context * bridge_ctx = ggml_init(bridge_params);
    if (!bridge_ctx) {
        ggml_backend_buffer_free(producer_buffer);
        ggml_backend_free(backend);
        ggml_free(producer_ctx);
        return 5;
    }

    std::string error;
    ggml_tensor * leaf =
        omni::flow::make_borrowed_device_leaf(
            bridge_ctx,
            producer_output,
            contract,
            strides,
            error);
    if (!leaf || !error.empty() || leaf->op != GGML_OP_NONE ||
        leaf->src[0] != nullptr ||
        leaf->data != producer_output->data ||
        leaf->buffer != producer_output->buffer) {
        ggml_free(bridge_ctx);
        ggml_backend_buffer_free(producer_buffer);
        ggml_backend_free(backend);
        ggml_free(producer_ctx);
        return 6;
    }

    ggml_tensor * tcb =
        ggml_cont(
            bridge_ctx,
            ggml_permute(bridge_ctx, leaf, 1, 0, 2, 3));
    ggml_tensor * destination =
        ggml_new_tensor_3d(
            bridge_ctx, GGML_TYPE_F32, 50, 80, 1);
    ggml_tensor * copy = ggml_cpy(bridge_ctx, tcb, destination);
    ggml_cgraph * bridge_graph =
        ggml_new_graph_custom(bridge_ctx, 64, false);
    ggml_build_forward_expand(bridge_graph, copy);
    if (ggml_graph_n_nodes(bridge_graph) > 8) {
        ggml_free(bridge_ctx);
        ggml_backend_buffer_free(producer_buffer);
        ggml_backend_free(backend);
        ggml_free(producer_ctx);
        return 7;
    }

    int64_t bad_strides[3] = {
        strides[0], strides[1] + 4, strides[2]};
    if (omni::flow::make_borrowed_device_leaf(
            bridge_ctx,
            producer_output,
            contract,
            bad_strides,
            error) != nullptr ||
        error.empty()) {
        ggml_free(bridge_ctx);
        ggml_backend_buffer_free(producer_buffer);
        ggml_backend_free(backend);
        ggml_free(producer_ctx);
        return 8;
    }

    ggml_free(bridge_ctx);
    ggml_backend_buffer_free(producer_buffer);
    ggml_backend_free(backend);
    ggml_free(producer_ctx);
    return 0;
}
