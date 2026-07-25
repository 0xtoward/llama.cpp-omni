#include "../tts-device-head.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "src/llama-graph.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef NDEBUG
#undef assert
#define assert(condition) do {                                                \
    if (!(condition)) {                                                       \
        std::cerr << "assertion failed: " #condition                          \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";           \
        std::abort();                                                         \
    }                                                                         \
} while (0)
#endif

static ggml_backend_t init_test_backend() {
    const char * requested = std::getenv("OMNI_TEST_TTS_BACKEND");
    if (requested && std::string(requested) == "cann") {
        ggml_backend_t backend =
                ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        assert(backend != nullptr);
        const std::string name = ggml_backend_name(backend);
        assert(name.find("CANN") != std::string::npos ||
               name.find("cann") != std::string::npos);
        return backend;
    }
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

static bool test_backend_is_cann(ggml_backend_t backend) {
    const std::string name = ggml_backend_name(backend);
    return name.find("CANN") != std::string::npos ||
           name.find("cann") != std::string::npos;
}

static void assert_production_transfer_contract(
        const omni::tts_device_head & runner,
        int hidden_size,
        bool stochastic) {
    const auto stats = runner.last_transfer_stats();
    assert(stats.production_contract_ok());
    assert(stats.d2h_bytes == sizeof(int32_t));
    assert(stats.control_scalar_d2h_bytes == sizeof(int32_t));
    assert(stats.d2d_bytes ==
           static_cast<uint64_t>(hidden_size) * sizeof(float));
    assert(stats.hidden_d2h_bytes == 0);
    assert(stats.logits_d2h_bytes == 0);
    assert(stats.embedding_h2d_bytes == 0);
    assert(stats.embedding_d2h_bytes == 0);
    assert(stats.diagnostic_d2h_bytes == 0);
    if (stochastic) {
        assert(stats.h2d_bytes > 0);
    } else {
        assert(stats.h2d_bytes == 0);
    }
}

static void test_config() {
    omni::tts_device_head_config config;
    std::string error;

    assert(omni::tts_device_head_parse_config(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, config, error));
    assert(config.mode == omni::tts_head_mode::cpu);
    assert(!config.device_sampler);
    assert(config.suppress_model_logits);
    assert(config.base_output == LLAMA_OUTPUT_DEFAULT);
    assert(config.device_burst == 1);

    assert(omni::tts_device_head_parse_config(
            "cann", "1", "1", "0", "full", "1", config, error));
    assert(config.mode == omni::tts_head_mode::cann);
    assert(config.device_sampler);
    assert(config.trace);
    assert(!config.suppress_model_logits);
    assert(config.base_output == LLAMA_OUTPUT_DEFAULT);

    assert(omni::tts_device_head_parse_config(
            "cann", "1", "0", "1", "hidden_only", "4", config, error));
    assert(config.base_output == LLAMA_OUTPUT_HIDDEN_ONLY);
    assert(config.device_burst == 4);

    assert(!omni::tts_device_head_parse_config(
            "cann", "0", nullptr, nullptr, nullptr, nullptr, config, error));
    assert(!error.empty());
    assert(!omni::tts_device_head_parse_config(
            "bogus", nullptr, nullptr, nullptr, nullptr, nullptr, config, error));
    assert(!omni::tts_device_head_parse_config(
            "cann", "1", nullptr, "bogus", nullptr, nullptr, config, error));
    assert(error == "OMNI_TTS_SUPPRESS_MODEL_LOGITS must be 0 or 1");
    assert(!omni::tts_device_head_parse_config(
            "cann", "1", nullptr, nullptr, "bogus", nullptr, config, error));
    assert(error == "OMNI_TTS_BASE_OUTPUT must be full or hidden_only");
    assert(!omni::tts_device_head_parse_config(
            "cpu", "0", nullptr, nullptr, "hidden_only", nullptr, config, error));
    assert(error == "OMNI_TTS_BASE_OUTPUT=hidden_only requires OMNI_TTS_HEAD=cann");
    assert(!omni::tts_device_head_parse_config(
            "cann", "1", nullptr, "0", "hidden_only", nullptr, config, error));
    assert(error ==
           "OMNI_TTS_BASE_OUTPUT=hidden_only conflicts with OMNI_TTS_SUPPRESS_MODEL_LOGITS=0");
    assert(!omni::tts_device_head_parse_config(
            "cann", "1", nullptr, "1", "hidden_only", "2", config, error));
    assert(error == "OMNI_TTS_DEVICE_BURST must be 1 or 4");
    assert(!omni::tts_device_head_parse_config(
            "cann", "1", nullptr, "1", "full", "4", config, error));
    assert(error ==
           "OMNI_TTS_DEVICE_BURST=4 requires OMNI_TTS_BASE_OUTPUT=hidden_only");
}

static void test_burst_eos_and_kv_rollback() {
    for (int eos_slot = 0; eos_slot < 4; ++eos_slot) {
        omni::tts_device_burst_state state(4);
        std::string error;
        assert(state.reset(7, error));
        assert(state.begin(7, error));
        for (int i = 0; i < 4; ++i) {
            assert(state.append(7, 100 + i, i == eos_slot, error));
        }

        omni::tts_device_burst_event event;
        assert(state.finish(
                7, /*keep_stop_embedding=*/false, event, error));
        assert(event.epoch == 7);
        assert(event.valid_count == 4);
        assert(event.accepted_count == eos_slot);
        assert(event.kv_rollback_count == 4 - eos_slot);
        assert(event.stops[eos_slot] == 1);
        const auto recent = state.recent_tokens();
        assert(static_cast<int>(recent.size()) == eos_slot);
        for (int i = 0; i < eos_slot; ++i) {
            assert(recent[i] == 100 + i);
        }
    }

    omni::tts_device_burst_state final_state(4);
    std::string error;
    assert(final_state.reset(9, error));
    assert(final_state.begin(9, error));
    for (int i = 0; i < 4; ++i) {
        assert(final_state.append(9, 200 + i, i == 2, error));
    }
    omni::tts_device_burst_event final_event;
    assert(final_state.finish(
            9, /*keep_stop_embedding=*/true, final_event, error));
    assert(final_event.accepted_count == 3);
    assert(final_event.kv_rollback_count == 1);
}

static void test_burst_epoch_reset_and_cancel() {
    omni::tts_device_burst_state state(4);
    std::string error;
    assert(state.reset(10, error));
    assert(state.begin(10, error));
    assert(state.append(10, 1, false, error));
    assert(state.cancel(11, error));
    assert(state.epoch() == 11);
    assert(state.valid_count() == 0);
    assert(state.recent_tokens().empty());

    assert(!state.append(10, 2, false, error));
    assert(error == "TTS device burst append rejected a stale epoch");
    assert(!state.begin(10, error));
    assert(error == "TTS device burst begin rejected a stale epoch");
    assert(!state.cancel(11, error));
    assert(error == "TTS device burst cancel requires a newer epoch");

    assert(state.begin(11, error));
    for (int i = 0; i < 4; ++i) {
        assert(state.append(11, 10 + i, false, error));
    }
    omni::tts_device_burst_event event;
    assert(!state.finish(10, false, event, error));
    assert(error == "TTS device burst finish rejected a stale epoch");
    assert(state.finish(11, false, event, error));
    assert(event.accepted_count == 4);
}

static void test_output_contract_participates_in_graph_reuse() {
    llm_graph_params full = {};
    llm_graph_params hidden_only = {};
    hidden_only.output_contract = LLAMA_OUTPUT_HIDDEN_ONLY;

    assert(full.allow_reuse(full));
    assert(hidden_only.allow_reuse(hidden_only));
    assert(!full.allow_reuse(hidden_only));
    assert(!hidden_only.allow_reuse(full));
}

static void test_greedy_device_graph() {
    constexpr int hidden_size = 4;
    constexpr int vocab_size = 8;

    ggml_backend_t backend = init_test_backend();
    assert(backend != nullptr);

    ggml_context_ptr hidden_ctx(ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    }));
    assert(hidden_ctx);
    ggml_tensor * hidden_tensor =
            ggml_new_tensor_2d(hidden_ctx.get(), GGML_TYPE_F32, hidden_size, 1);
    ggml_backend_buffer_ptr hidden_buffer(
            ggml_backend_alloc_ctx_tensors(hidden_ctx.get(), backend));
    assert(hidden_buffer);

    const float hidden_data[hidden_size] = { 1.0f, 2.0f, -1.0f, 0.5f };
    ggml_backend_tensor_set(hidden_tensor, hidden_data, 0, sizeof(hidden_data));

    // Rows are logical audio-code entries because GGML sees [hidden, vocab].
    std::vector<float> head(hidden_size * vocab_size, 0.0f);
    head[3 * hidden_size + 0] = 10.0f;
    std::vector<float> emb(hidden_size * vocab_size, 0.0f);
    for (int token = 0; token < vocab_size; ++token) {
        for (int d = 0; d < hidden_size; ++d) {
            emb[token * hidden_size + d] = token * 10.0f + d;
        }
    }

    llama_device_tensor hidden = {};
    hidden.tensor = hidden_tensor;
    hidden.backend = backend;
    hidden.type = GGML_TYPE_F32;
    hidden.ne[0] = hidden_size;
    hidden.ne[1] = 1;

    omni::tts_device_head runner;
    std::string error;
    const bool initialized = runner.initialize(
            hidden,
            head.data(),
            emb.data(),
            hidden_size,
            vocab_size,
            /*temperature=*/0.8f,
            /*top_p=*/0.85f,
            /*top_k=*/4,
            /*min_keep=*/2,
            /*repetition_penalty=*/1.05f,
            /*repetition_window=*/4,
            /*greedy=*/true,
            /*apply_top_k_p=*/true,
            /*require_cann=*/test_backend_is_cann(backend),
            /*trace=*/false,
            error);
    if (!initialized) {
        std::cerr << "TTS greedy device head initialization failed: "
                  << error << "\n";
    }
    assert(initialized);

    omni::tts_device_head_step step;
    step.skip_repetition = true;
    int32_t token = -1;
    llama_device_tensor selected_embedding = {};
    assert(runner.forward(hidden, step, token, selected_embedding, error));
    // Validate the runner's own hot path before this test intentionally reads
    // the selected embedding back for numerical comparison.
    assert_production_transfer_contract(
            runner, hidden_size, /*stochastic=*/false);
    assert(token == 3);
    assert(selected_embedding.tensor != nullptr);
    assert(selected_embedding.backend == backend);
    assert(selected_embedding.ne[0] == hidden_size);
    assert(selected_embedding.ne[1] == 1);

    std::vector<float> got(hidden_size);
    ggml_backend_tensor_get(
            static_cast<ggml_tensor *>(selected_embedding.tensor),
            got.data(), 0, got.size() * sizeof(float));
    for (int d = 0; d < hidden_size; ++d) {
        assert(std::fabs(got[d] - (30.0f + d)) < 1e-6f);
    }

    assert(setenv("OMNI_TTS_HIDDEN_FINGERPRINT", "1", 1) == 0);
    step.token_index = 7;
    step.n_past = 31;
    assert(runner.forward(hidden, step, token, selected_embedding, error));
    const auto diagnostic_stats = runner.last_transfer_stats();
    assert(diagnostic_stats.hidden_d2h_bytes == sizeof(hidden_data));
    assert(diagnostic_stats.diagnostic_d2h_bytes == sizeof(hidden_data));
    assert(diagnostic_stats.d2h_bytes ==
           sizeof(hidden_data) + sizeof(int32_t));
    assert(unsetenv("OMNI_TTS_HIDDEN_FINGERPRINT") == 0);

    runner.reset();
    hidden_buffer.reset();
    hidden_ctx.reset();
    ggml_backend_free(backend);
}

static int cpu_reference_sample(
        const std::vector<float> & base_logits,
        const std::vector<int32_t> & recent,
        int repetition_window,
        float repetition_penalty,
        float temperature,
        bool skip_repetition,
        bool force_no_eos,
        bool apply_top_k_p,
        int top_k,
        float top_p,
        int min_keep,
        float uniform) {
    std::vector<float> logits = base_logits;
    for (float & value : logits) {
        value /= temperature;
    }

    if (!skip_repetition) {
        std::unordered_map<int32_t, int32_t> frequencies;
        const size_t begin = recent.size() > static_cast<size_t>(repetition_window)
                ? recent.size() - repetition_window : 0;
        for (size_t i = begin; i < recent.size(); ++i) {
            frequencies[recent[i]]++;
        }
        for (const auto & [token, frequency] : frequencies) {
            const float alpha = std::pow(repetition_penalty, frequency);
            logits[token] = logits[token] < 0.0f
                    ? logits[token] * alpha : logits[token] / alpha;
        }
    }
    if (force_no_eos) {
        logits.back() = -INFINITY;
    }

    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (float & probability : probs) {
        probability /= sum;
    }

    std::vector<int32_t> candidates(logits.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        candidates[i] = static_cast<int32_t>(i);
    }
    if (apply_top_k_p) {
        std::stable_sort(
                candidates.begin(), candidates.end(),
                [&probs](int32_t lhs, int32_t rhs) {
                    return probs[lhs] > probs[rhs];
                });
        std::vector<int32_t> filtered;
        float cumulative = 0.0f;
        for (int32_t token : candidates) {
            if (static_cast<int>(filtered.size()) < min_keep ||
                (cumulative < top_p && static_cast<int>(filtered.size()) < top_k)) {
                cumulative += probs[token];
                filtered.push_back(token);
            } else {
                break;
            }
        }
        candidates = std::move(filtered);
    }

    float selected_sum = 0.0f;
    for (int32_t token : candidates) {
        selected_sum += probs[token];
    }
    float cumulative = 0.0f;
    for (int32_t token : candidates) {
        cumulative += probs[token] / selected_sum;
        if (uniform <= cumulative) {
            return token;
        }
    }
    return candidates.back();
}

static void test_fixed_uniform_32_codes(bool apply_top_k_p) {
    constexpr int hidden_size = 4;
    constexpr int vocab_size = 32;
    constexpr int repetition_window = 8;
    constexpr float repetition_penalty = 1.05f;
    constexpr float temperature = 0.8f;
    constexpr float top_p = 0.85f;
    constexpr int top_k = 10;
    constexpr int min_keep = 3;

    ggml_backend_t backend = init_test_backend();
    assert(backend != nullptr);
    ggml_context_ptr hidden_ctx(ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    }));
    ggml_tensor * hidden_tensor =
            ggml_new_tensor_2d(hidden_ctx.get(), GGML_TYPE_F32, hidden_size, 1);
    ggml_backend_buffer_ptr hidden_buffer(
            ggml_backend_alloc_ctx_tensors(hidden_ctx.get(), backend));
    assert(hidden_buffer);
    const float hidden_data[hidden_size] = { 0.75f, -0.5f, 1.25f, 0.375f };
    ggml_backend_tensor_set(hidden_tensor, hidden_data, 0, sizeof(hidden_data));

    std::vector<float> head(hidden_size * vocab_size);
    std::vector<float> emb(hidden_size * vocab_size);
    std::vector<float> base_logits(vocab_size, 0.0f);
    for (int token = 0; token < vocab_size; ++token) {
        for (int d = 0; d < hidden_size; ++d) {
            const float value =
                    std::sin((token + 1) * (d + 2) * 0.37f) +
                    0.013f * token - 0.021f * d;
            head[token * hidden_size + d] = value;
            emb[token * hidden_size + d] = token + d * 0.1f;
            base_logits[token] += value * hidden_data[d];
        }
    }

    llama_device_tensor hidden = {};
    hidden.tensor = hidden_tensor;
    hidden.backend = backend;
    hidden.type = GGML_TYPE_F32;
    hidden.ne[0] = hidden_size;
    hidden.ne[1] = 1;

    omni::tts_device_head runner;
    std::string error;
    const bool initialized = runner.initialize(
            hidden,
            head.data(),
            emb.data(),
            hidden_size,
            vocab_size,
            temperature,
            top_p,
            top_k,
            min_keep,
            repetition_penalty,
            repetition_window,
            /*greedy=*/false,
            apply_top_k_p,
            /*require_cann=*/test_backend_is_cann(backend),
            /*trace=*/false,
            error);
    if (!initialized) {
        std::cerr << "TTS stochastic device head initialization failed: "
                  << error << " (top_k_p=" << apply_top_k_p << ")\n";
    }
    assert(initialized);

    std::vector<int32_t> recent;
    omni::tts_device_burst_state burst_state(4);
    assert(burst_state.reset(32, error));
    for (int step_idx = 0; step_idx < 32; ++step_idx) {
        if (step_idx % 4 == 0) {
            assert(burst_state.begin(32, error));
        }
        const float uniform =
                (static_cast<float>((step_idx * 37) % 97) + 0.25f) / 98.0f;
        omni::tts_device_head_step step;
        step.recent_relative_tokens = recent;
        step.skip_repetition = step_idx == 0;
        step.force_no_eos = step_idx < 5;
        step.has_uniform = true;
        step.uniform = uniform;

        const int expected = cpu_reference_sample(
                base_logits,
                recent,
                repetition_window,
                repetition_penalty,
                temperature,
                step.skip_repetition,
                step.force_no_eos,
                apply_top_k_p,
                top_k,
                top_p,
                min_keep,
                uniform);
        int32_t actual = -1;
        llama_device_tensor selected_embedding = {};
        assert(runner.forward(hidden, step, actual, selected_embedding, error));
        assert_production_transfer_contract(
                runner, hidden_size, /*stochastic=*/true);
        if (actual != expected) {
            std::cerr << "fixed-uniform mismatch at step " << step_idx
                      << ": actual=" << actual << " expected=" << expected
                      << " top_k_p=" << apply_top_k_p << "\n";
            std::abort();
        }
        recent.push_back(actual);
        assert(burst_state.append(
                32, actual, /*stop=*/false, error));
        if (step_idx % 4 == 3) {
            omni::tts_device_burst_event event;
            assert(burst_state.finish(
                    32, /*keep_stop_embedding=*/false, event, error));
            assert(event.valid_count == 4);
            assert(event.accepted_count == 4);
            assert(event.kv_rollback_count == 0);
            const auto ring_recent = burst_state.recent_tokens();
            const size_t expected_size =
                    std::min<size_t>(
                            recent.size(),
                            omni::tts_device_burst_state::recent_capacity);
            assert(ring_recent.size() == expected_size);
            assert(std::equal(
                    ring_recent.begin(),
                    ring_recent.end(),
                    recent.end() - expected_size));
        }
    }

    runner.reset();
    hidden_buffer.reset();
    hidden_ctx.reset();
    ggml_backend_free(backend);
}

static void test_no_d2h_four_step_primitive() {
    constexpr int hidden_size = 4;
    constexpr int vocab_size = 16;
    constexpr float temperature = 0.8f;

    ggml_backend_t backend = init_test_backend();
    assert(backend != nullptr);
    ggml_context_ptr hidden_ctx(ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    }));
    ggml_tensor * hidden_tensor =
            ggml_new_tensor_2d(
                    hidden_ctx.get(), GGML_TYPE_F32, hidden_size, 1);
    ggml_backend_buffer_ptr hidden_buffer(
            ggml_backend_alloc_ctx_tensors(hidden_ctx.get(), backend));
    assert(hidden_buffer);
    const float hidden_data[hidden_size] = {
        0.125f, -0.75f, 1.5f, 0.625f,
    };
    ggml_backend_tensor_set(
            hidden_tensor, hidden_data, 0, sizeof(hidden_data));

    std::vector<float> head(hidden_size * vocab_size);
    std::vector<float> emb(hidden_size * vocab_size);
    std::vector<float> logits(vocab_size, 0.0f);
    for (int token = 0; token < vocab_size; ++token) {
        for (int d = 0; d < hidden_size; ++d) {
            const float value =
                    std::cos((token + 2) * (d + 1) * 0.19f) +
                    0.017f * token;
            head[token * hidden_size + d] = value;
            emb[token * hidden_size + d] = token + 0.01f * d;
            logits[token] += value * hidden_data[d];
        }
    }

    llama_device_tensor hidden = {};
    hidden.tensor = hidden_tensor;
    hidden.backend = backend;
    hidden.type = GGML_TYPE_F32;
    hidden.ne[0] = hidden_size;
    hidden.ne[1] = 1;

    omni::tts_device_head runner;
    std::string error;
    assert(runner.initialize(
            hidden,
            head.data(),
            emb.data(),
            hidden_size,
            vocab_size,
            temperature,
            /*top_p=*/0.85f,
            /*top_k=*/8,
            /*min_keep=*/3,
            /*repetition_penalty=*/1.05f,
            /*repetition_window=*/8,
            /*greedy=*/false,
            /*apply_top_k_p=*/false,
            /*require_cann=*/test_backend_is_cann(backend),
            /*trace=*/false,
            error));

    const std::array<float, 4> uniforms = {
        0.05f, 0.31f, 0.67f, 0.91f,
    };
    std::array<int32_t, 4> expected = {};
    for (int i = 0; i < 4; ++i) {
        expected[i] = cpu_reference_sample(
                logits,
                /*recent=*/{},
                /*repetition_window=*/8,
                /*repetition_penalty=*/1.05f,
                temperature,
                /*skip_repetition=*/true,
                /*force_no_eos=*/false,
                /*apply_top_k_p=*/false,
                /*top_k=*/8,
                /*top_p=*/0.85f,
                /*min_keep=*/3,
                uniforms[i]);
    }

    assert(runner.begin_burst(44, uniforms, error));
    for (int i = 0; i < 4; ++i) {
        omni::tts_device_head_step step;
        step.skip_repetition = true;
        step.has_uniform = true;
        step.uniform = uniforms[i];
        llama_device_tensor embedding = {};
        assert(runner.forward_burst_step(
                44, hidden, step, embedding, error));
        assert(embedding.tensor != nullptr);
        assert(runner.last_transfer_stats().d2h_bytes == 0);
        assert(runner.last_transfer_stats().control_scalar_d2h_bytes == 0);
    }
    omni::tts_device_burst_event event;
    assert(runner.finish_burst(44, event, error));
    assert(event.valid_count == 4);
    assert(event.tokens == expected);
    assert(runner.last_transfer_stats().d2h_bytes ==
           4 * sizeof(int32_t));
    assert(runner.last_transfer_stats().control_scalar_d2h_bytes ==
           4 * sizeof(int32_t));
    assert(!runner.finish_burst(44, event, error));

    assert(runner.begin_burst(45, uniforms, error));
    omni::tts_device_head_step exact_step;
    exact_step.has_uniform = true;
    exact_step.uniform = uniforms[0];
    exact_step.skip_repetition = false;
    llama_device_tensor unused_embedding = {};
    assert(!runner.forward_burst_step(
            45, hidden, exact_step, unused_embedding, error));
    assert(error.find("device-side repetition history") !=
           std::string::npos);
    runner.cancel_burst(46);

    runner.reset();
    hidden_buffer.reset();
    hidden_ctx.reset();
    ggml_backend_free(backend);
}

static void test_kv_suffix_rollback_contract() {
    std::string error;
    int32_t n_past = 20;
    bool called = false;
    assert(omni::tts_device_burst_rollback_kv(
            n_past,
            3,
            [&](int32_t p0, int32_t p1) {
                called = true;
                assert(p0 == 17);
                assert(p1 == 20);
                return true;
            },
            error));
    assert(called);
    assert(n_past == 17);

    n_past = 20;
    assert(!omni::tts_device_burst_rollback_kv(
            n_past,
            4,
            [](int32_t, int32_t) { return false; },
            error));
    assert(n_past == 20);
    assert(error == "llama_memory_seq_rm rejected the TTS KV suffix");
    assert(!omni::tts_device_burst_rollback_kv(
            n_past,
            21,
            [](int32_t, int32_t) { return true; },
            error));
    assert(n_past == 20);
}

int main() {
    test_config();
    test_burst_eos_and_kv_rollback();
    test_burst_epoch_reset_and_cancel();
    test_output_contract_participates_in_graph_reuse();
    test_greedy_device_graph();
    test_fixed_uniform_32_codes(/*apply_top_k_p=*/false);
    test_fixed_uniform_32_codes(/*apply_top_k_p=*/true);
    test_no_d2h_four_step_primitive();
    test_kv_suffix_rollback_contract();
    std::cout << "TTS device head tests passed\n";
    return 0;
}
