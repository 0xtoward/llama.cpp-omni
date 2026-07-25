#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace omni::flow {

struct device_bridge_config {
    bool        enabled = false;
    std::string error;

    explicit operator bool() const {
        return error.empty();
    }
};

inline device_bridge_config parse_device_bridge_config(const char * raw) {
    device_bridge_config result;
    const std::string value = raw == nullptr ? "0" : raw;
    if (value == "0") {
        return result;
    }
    if (value == "1") {
        result.enabled = true;
        return result;
    }
    result.error = "OMNI_T2W_DEVICE_BRIDGE must be 0|1";
    return result;
}

inline device_bridge_config device_bridge_config_from_environment() {
    return parse_device_bridge_config(
        std::getenv("OMNI_T2W_DEVICE_BRIDGE"));
}

enum class device_bridge_dtype : uint8_t {
    f32,
    unsupported,
};

struct borrowed_device_tensor_contract {
    uintptr_t source_device = 0;
    uintptr_t data = 0;
    int64_t   channels = 0;
    int64_t   frames = 0;
    int64_t   batch = 0;
    size_t    bytes = 0;
    device_bridge_dtype dtype = device_bridge_dtype::unsupported;
    bool      producer_synchronized = false;
    uint64_t  epoch = 0;
};

inline std::string validate_borrowed_device_tensor_contract(
        const borrowed_device_tensor_contract & contract,
        uintptr_t expected_device,
        uint64_t expected_epoch) {
    if (contract.source_device == 0 || contract.data == 0) {
        return "device bridge requires a live device tensor";
    }
    if (contract.source_device != expected_device) {
        return "device bridge requires Token2Mel and HiFT on the same device";
    }
    if (!contract.producer_synchronized) {
        return "device bridge requires a producer event or explicit synchronization";
    }
    if (expected_epoch != 0 && contract.epoch != expected_epoch) {
        return "device bridge rejected a stale session epoch";
    }
    if (contract.dtype != device_bridge_dtype::f32) {
        return "device bridge requires F32 mel";
    }
    if (contract.channels != 80 || contract.frames <= 0 ||
        contract.batch != 1) {
        return "device bridge requires CTB shape [80,T,1]";
    }
    const size_t expected_bytes =
        static_cast<size_t>(contract.channels) *
        static_cast<size_t>(contract.frames) *
        static_cast<size_t>(contract.batch) * sizeof(float);
    if (contract.bytes != expected_bytes) {
        return "device bridge byte count does not match shape";
    }
    return {};
}

}  // namespace omni::flow
