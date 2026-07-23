#pragma once

#include <cstdlib>
#include <string>

namespace omni::flow {

inline bool token2wav_require_npu() {
    const char * value = std::getenv("OMNI_REQUIRE_NPU_T2W");
    return value != nullptr && std::string(value) == "1";
}

inline bool token2wav_accelerator_device_requested(const std::string & device) {
    return device == "gpu" || device.rfind("gpu:", 0) == 0;
}

inline bool token2wav_backend_name_is_cann(const std::string & backend_name) {
    return backend_name.rfind("CANN", 0) == 0;
}

}  // namespace omni::flow
