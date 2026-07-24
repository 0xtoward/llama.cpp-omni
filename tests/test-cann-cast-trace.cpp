#include "../ggml/src/ggml-cann/cast-trace.h"

#include <cassert>
#include <stdexcept>
#include <string>

using namespace ggml_cann_cast_trace;

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "configured") {
        const mode expected = parse_mode(argv[2]);
        assert(configured_mode() == expected);

        event configured_value;
        configured_value.cast_domain = domain::thinker;
        configured_value.cast_origin = origin::fia;
        configured_value.producer = "q";
        configured_value.consumer = "fattn";
        configured_value.src_dtype = "f32";
        configured_value.dst_dtype = "f16";
        configured_value.shape = {128, 32, 1, 1};
        configured_value.bytes = 8192;
        record(configured_value, 1234);
        flush_summary();
        return 0;
    }

    assert(parse_mode("") == mode::off);
    assert(parse_mode("off") == mode::off);
    assert(parse_mode("summary") == mode::summary);
    assert(parse_mode("jsonl") == mode::jsonl);

    bool invalid_rejected = false;
    try {
        (void) parse_mode("true");
    } catch (const std::invalid_argument &) {
        invalid_rejected = true;
    }
    assert(invalid_rejected);

    assert(infer_domain("cache_k_l0", "cache_v_l0") == domain::thinker);
    assert(infer_domain("MiniCPMTTS", "talker") == domain::tts);
    assert(infer_domain("token2mel.block0", "") == domain::token2mel);
    assert(infer_domain("HiFT.generator", "") == domain::hift);
    assert(infer_domain("unknown", "cast") == domain::generic);

    event value;
    value.cast_domain = domain::thinker;
    value.cast_origin = origin::kv;
    value.producer = "k\\\"cur";
    value.consumer = "cache_k_l0";
    value.src_dtype = "f32";
    value.dst_dtype = "f16";
    value.shape = {128, 8, 1, 1};
    value.bytes = 2048;

    const std::string json = event_json(value, 4, 8192, 1000);
    assert(json.find("\"domain\":\"thinker\"") != std::string::npos);
    assert(json.find("\"origin\":\"kv\"") != std::string::npos);
    assert(json.find("\"producer\":\"k\\\\\\\"cur\"") != std::string::npos);
    assert(json.find("\"shape\":[128,8,1,1]") != std::string::npos);
    assert(json.find("\"calls\":4") != std::string::npos);
    assert(json.find("\"bytes\":8192") != std::string::npos);
    assert(json.find("\"host_submit_ns_avg\":250") != std::string::npos);
    assert(json.find("\"host_api_time_ns_avg\":250") != std::string::npos);
    assert(json.find("\"host_api_time_source\":"
                     "\"aclnn_workspace_and_async_launch_wall\"") !=
           std::string::npos);
    assert(json.find("\"device_time_ns\":null") != std::string::npos);
    assert(json.find("\"device_time_source\":\"msprof_postprocess\"") !=
           std::string::npos);
    return 0;
}
