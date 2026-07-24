#include "token-trace.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace fs = std::filesystem;

static void write_text(const fs::path & path, const std::string & text) {
    std::ofstream out(path);
    out << text;
}

int main() {
    const fs::path root = fs::temp_directory_path() / "omni-token-trace-test";
    fs::remove_all(root);
    fs::create_directories(root);

    std::string valid;
    for (int i = 0; i < 32; ++i) {
        valid += std::to_string(i) + (i % 4 == 3 ? "\n" : " ");
    }
    write_text(root / "valid.tokens", valid);
    write_text(root / "short.tokens", "0 1 2\n");
    write_text(root / "negative.tokens", "-1\n");
    write_text(root / "junk.tokens", "0 1 2x\n");
    write_text(root / "overflow.tokens",
               "2147483648 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 "
               "16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\n");

    std::string error;
    std::vector<int32_t> tokens;
    assert(omni::token_trace::parse_teacher_tokens_file(
        (root / "valid.tokens").string(), tokens, error));
    assert(tokens.size() == 32 && tokens.front() == 0 && tokens.back() == 31);
    assert(!omni::token_trace::parse_teacher_tokens_file(
        (root / "short.tokens").string(), tokens, error));
    assert(!omni::token_trace::parse_teacher_tokens_file(
        (root / "negative.tokens").string(), tokens, error));
    assert(!omni::token_trace::parse_teacher_tokens_file(
        (root / "junk.tokens").string(), tokens, error));
    assert(!omni::token_trace::parse_teacher_tokens_file(
        (root / "overflow.tokens").string(), tokens, error));

    omni::token_trace::state disabled;
    error.clear();
    assert(omni::token_trace::initialize(nullptr, nullptr, disabled, error));
    assert(!disabled.enabled());
    assert(!omni::token_trace::initialize(
        nullptr, (root / "valid.tokens").c_str(), disabled, error));

    omni::token_trace::state debug;
    error.clear();
    const std::string trace_path = (root / "trace.jsonl").string();
    const std::string teacher_path = (root / "valid.tokens").string();
    assert(omni::token_trace::initialize(
        trace_path.c_str(), teacher_path.c_str(), debug, error));
    assert(debug.enabled() && debug.teacher_enabled());
    for (int i = 0; i < 32; ++i) {
        int32_t selected = -1;
        size_t index = 99;
        assert(omni::token_trace::next_teacher_token(debug, selected, index, error));
        assert(selected == i && index == static_cast<size_t>(i));
    }
    int32_t exhausted = -1;
    size_t exhausted_index = 0;
    assert(!omni::token_trace::next_teacher_token(
        debug, exhausted, exhausted_index, error));

    const float logits[] = {
        1.0f, 3.0f, 3.0f, std::numeric_limits<float>::quiet_NaN(), -2.0f,
    };
    const auto top = omni::token_trace::compute_top_k(logits, 5, 3);
    assert(top.size() == 3);
    assert(top[0].token == 1 && top[1].token == 2 && top[2].token == 0);

    const float hidden[] = {
        1.0f, -2.0f, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    const auto fingerprint = omni::token_trace::fingerprint(hidden, 4);
    assert(fingerprint.count == 4);
    assert(fingerprint.finite_count == 2);
    assert(fingerprint.inf_count == 1);
    assert(fingerprint.nan_count == 1);
    assert(fingerprint.min == -2.0f && fingerprint.max == 1.0f);
    assert(std::abs(fingerprint.mean + 0.5) < 1e-12);

    omni::token_trace::record event;
    event.step = 7;
    event.position = 41;
    event.n_past_before = 41;
    event.n_past_after = 42;
    event.turn_id = 3;
    event.session_generation = 2;
    event.prompt_token_count = 41;
    event.request_prefill_token_count = 9;
    event.session_hint = "session-\"a\"\n";
    event.greedy_token = 11;
    event.sampled_token = 11;
    event.selected_token = 17;
    event.token_type = "NORMAL";
    event.teacher_active = true;
    event.teacher_index = 7;
    event.raw_logits_available = true;
    event.top_k = top;
    event.hidden_available = true;
    event.hidden = fingerprint;
    const std::string json = omni::token_trace::to_json(event);
    assert(json.find("\"event\":\"omni_token_trace\"") != std::string::npos);
    assert(json.find("\"selected_token\":17") != std::string::npos);
    assert(json.find("\"session_generation\":2") != std::string::npos);
    assert(json.find("\"prompt_token_count\":41") != std::string::npos);
    assert(json.find("\"request_prefill_token_count\":9") != std::string::npos);
    assert(json.find("\"session_hint\":\"session-\\\"a\\\"\\n\"") != std::string::npos);
    assert(json.find("\"top_k\":[{\"token\":1") != std::string::npos);
    assert(json.find("\"hidden\":{\"count\":4") != std::string::npos);
    omni::token_trace::record non_finite_event;
    non_finite_event.raw_logits_available = true;
    non_finite_event.top_k = {{1, -std::numeric_limits<float>::infinity()}};
    const std::string non_finite_json =
        omni::token_trace::to_json(non_finite_event);
    assert(non_finite_json.find("\"logit\":null") != std::string::npos);
    assert(omni::token_trace::append_jsonl(debug, event, error));

    std::ifstream trace(trace_path);
    std::string line;
    assert(std::getline(trace, line));
    assert(line == json);

    fs::remove_all(root);
    return 0;
}
