#include "../tts-device-head.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

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

static void test_config() {
    omni::tts_device_head_config config;
    std::string error;

    assert(omni::tts_device_head_parse_config(nullptr, nullptr, nullptr, config, error));
    assert(config.mode == omni::tts_head_mode::cpu);
    assert(!config.device_sampler);

    assert(omni::tts_device_head_parse_config("cann", "1", "1", config, error));
    assert(config.mode == omni::tts_head_mode::cann);
    assert(config.device_sampler);
    assert(config.trace);

    assert(!omni::tts_device_head_parse_config("cann", "0", nullptr, config, error));
    assert(!error.empty());
    assert(!omni::tts_device_head_parse_config("bogus", nullptr, nullptr, config, error));
}

static void test_greedy_device_graph() {
    constexpr int hidden_size = 4;
    constexpr int vocab_size = 8;

    ggml_backend_t backend =
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
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
    assert(runner.initialize(
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
            /*require_cann=*/false,
            /*trace=*/false,
            error));

    omni::tts_device_head_step step;
    step.skip_repetition = true;
    int32_t token = -1;
    llama_device_tensor selected_embedding = {};
    assert(runner.forward(hidden, step, token, selected_embedding, error));
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

    ggml_backend_t backend =
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
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
    assert(runner.initialize(
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
            /*require_cann=*/false,
            /*trace=*/false,
            error));

    std::vector<int32_t> recent;
    for (int step_idx = 0; step_idx < 32; ++step_idx) {
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
        if (actual != expected) {
            std::cerr << "fixed-uniform mismatch at step " << step_idx
                      << ": actual=" << actual << " expected=" << expected
                      << " top_k_p=" << apply_top_k_p << "\n";
            std::abort();
        }
        recent.push_back(actual);
    }

    runner.reset();
    hidden_buffer.reset();
    hidden_ctx.reset();
    ggml_backend_free(backend);
}

int main() {
    test_config();
    test_greedy_device_graph();
    test_fixed_uniform_32_codes(/*apply_top_k_p=*/false);
    test_fixed_uniform_32_codes(/*apply_top_k_p=*/true);
    std::cout << "TTS device head tests passed\n";
    return 0;
}
