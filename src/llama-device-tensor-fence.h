#pragma once

#include <string>

enum class llama_device_tensor_fence_mode {
    sync,
    backend_sync,
    event_sync,
    stream_wait,
};

struct llama_device_tensor_fence_config {
    llama_device_tensor_fence_mode mode =
            llama_device_tensor_fence_mode::sync;
    std::string error;

    explicit operator bool() const {
        return error.empty();
    }
};

inline llama_device_tensor_fence_config
llama_parse_device_tensor_fence_config(const char * raw) {
    llama_device_tensor_fence_config result;
    const std::string value = raw ? raw : "sync";
    if (value == "sync") {
        result.mode = llama_device_tensor_fence_mode::sync;
    } else if (value == "backend_sync") {
        result.mode = llama_device_tensor_fence_mode::backend_sync;
    } else if (value == "event_sync") {
        result.mode = llama_device_tensor_fence_mode::event_sync;
    } else if (value == "stream_wait") {
        result.mode = llama_device_tensor_fence_mode::stream_wait;
    } else {
        result.error =
                "OMNI_TTS_HIDDEN_FENCE must be "
                "sync|backend_sync|event_sync|stream_wait";
    }
    return result;
}
