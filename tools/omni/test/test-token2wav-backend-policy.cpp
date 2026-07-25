#include "token2wav/token2wav-backend-policy.h"
#include "token2wav/token2wav-device-bridge-policy.h"

#include <cstdlib>

static void set_strict_env(const char * value) {
#ifdef _WIN32
    _putenv_s("OMNI_REQUIRE_NPU_T2W", value);
#else
    setenv("OMNI_REQUIRE_NPU_T2W", value, 1);
#endif
}

int main() {
    set_strict_env("0");
    if (omni::flow::token2wav_require_npu()) {
        return 1;
    }
    set_strict_env("1");
    if (!omni::flow::token2wav_require_npu()) {
        return 2;
    }
    if (!omni::flow::token2wav_accelerator_device_requested("gpu") ||
        !omni::flow::token2wav_accelerator_device_requested("gpu:1") ||
        omni::flow::token2wav_accelerator_device_requested("cpu")) {
        return 3;
    }
    if (!omni::flow::token2wav_backend_name_is_cann("CANN0") ||
        omni::flow::token2wav_backend_name_is_cann("CPU")) {
        return 4;
    }
    {
        const auto disabled = omni::flow::parse_device_bridge_config(nullptr);
        const auto enabled = omni::flow::parse_device_bridge_config("1");
        if (!disabled || disabled.enabled || !enabled || !enabled.enabled ||
            omni::flow::parse_device_bridge_config("") ||
            omni::flow::parse_device_bridge_config("true") ||
            omni::flow::parse_device_bridge_config("2")) {
            return 5;
        }
    }
    {
        omni::flow::borrowed_device_tensor_contract contract;
        contract.source_device = 0x1000;
        contract.data = 0x2000;
        contract.channels = 80;
        contract.frames = 50;
        contract.batch = 1;
        contract.bytes = 80 * 50 * sizeof(float);
        contract.dtype = omni::flow::device_bridge_dtype::f32;
        contract.producer_synchronized = true;
        contract.epoch = 7;
        if (!omni::flow::validate_borrowed_device_tensor_contract(
                 contract, 0x1000, 7).empty()) {
            return 6;
        }
        auto invalid = contract;
        invalid.source_device = 0x1001;
        if (omni::flow::validate_borrowed_device_tensor_contract(
                invalid, 0x1000, 7).empty()) {
            return 7;
        }
        invalid = contract;
        invalid.epoch = 6;
        if (omni::flow::validate_borrowed_device_tensor_contract(
                invalid, 0x1000, 7).empty()) {
            return 8;
        }
        invalid = contract;
        invalid.producer_synchronized = false;
        if (omni::flow::validate_borrowed_device_tensor_contract(
                invalid, 0x1000, 7).empty()) {
            return 9;
        }
        invalid = contract;
        invalid.dtype = omni::flow::device_bridge_dtype::unsupported;
        if (omni::flow::validate_borrowed_device_tensor_contract(
                invalid, 0x1000, 7).empty()) {
            return 10;
        }
        invalid = contract;
        invalid.channels = 79;
        if (omni::flow::validate_borrowed_device_tensor_contract(
                invalid, 0x1000, 7).empty()) {
            return 11;
        }
        invalid = contract;
        invalid.bytes -= sizeof(float);
        if (omni::flow::validate_borrowed_device_tensor_contract(
                invalid, 0x1000, 7).empty()) {
            return 12;
        }
    }
    return 0;
}
