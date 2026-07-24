#include "../tts-device-head.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
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
            /*seed=*/20260724,
            /*greedy=*/true,
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

int main() {
    test_config();
    test_greedy_device_graph();
    std::cout << "TTS device head tests passed\n";
    return 0;
}
