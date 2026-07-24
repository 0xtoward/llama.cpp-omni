#include "../e2e-trace.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

static std::string read_all(const std::filesystem::path & path) {
    std::ifstream input(path);
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

int main() {
    const auto root =
        std::filesystem::temp_directory_path() / "omni-e2e-trace-test";
    std::filesystem::create_directories(root);
    const auto trace_path = root / "trace.jsonl";
    const auto tensor_path = root / "tensor.jsonl";

    omni::e2e_trace::state trace;
    std::string error;
    assert(trace.initialize(
        "jsonl",
        trace_path.c_str(),
        "jsonl",
        tensor_path.c_str(),
        error));

    omni::e2e_trace::context ids;
    ids.session_id = "session-\"quoted";
    ids.epoch = 2;
    ids.turn_id = 3;
    ids.frame_id = 4;
    ids.chunk_id = 5;
    ids.queue_depth = 6;

    assert(trace.append_span(
        ids, "tts", "head.matmul", "npu:0", "default", 100, 250));
    assert(trace.append_instant(ids, "tts", "first_code", "npu:0"));

    omni::e2e_trace::tensor_record tensor;
    tensor.ids = ids;
    tensor.stage = "tts";
    tensor.action = "embedding.gather";
    tensor.tensor_role = "tts_next_embedding";
    tensor.producer = "device_gather";
    tensor.consumer = "tts_decode";
    tensor.shape = "[1,768]";
    tensor.dtype = "f32";
    tensor.device = "npu:0";
    tensor.stream = "default";
    tensor.transfer = "device_to_device";
    tensor.residency_claim = "host_roundtrip_removed";
    tensor.bytes = 768 * sizeof(float);
    tensor.buffer_id = 0x1234;
    tensor.reused = true;
    assert(trace.append_tensor(tensor));

    const std::string trace_text = read_all(trace_path);
    assert(trace_text.find("\"schema_version\":\"minicpmo45-omni-trace-event/v1\"") !=
           std::string::npos);
    assert(trace_text.find("\"name\":\"tts.head.matmul\"") !=
           std::string::npos);
    assert(trace_text.find("session-\\\"quoted") != std::string::npos);

    const std::string tensor_text = read_all(tensor_path);
    assert(tensor_text.find("\"transfer\":\"device_to_device\"") !=
           std::string::npos);
    assert(tensor_text.find("\"buffer_id\":\"0x1234\"") !=
           std::string::npos);
    assert(tensor_text.find("\"reused\":true") != std::string::npos);

    omni::e2e_trace::state invalid;
    assert(!invalid.initialize(
        "jsonl", nullptr, "off", nullptr, error));
    assert(error.find("OMNI_TRACE_OUTPUT") != std::string::npos);

    std::filesystem::remove_all(root);
    return 0;
}
