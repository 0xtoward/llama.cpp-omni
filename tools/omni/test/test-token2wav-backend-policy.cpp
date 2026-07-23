#include "token2wav/token2wav-backend-policy.h"

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
    return 0;
}
