#include "tts-device-head.h"

#include "common/log.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>

namespace omni {

namespace {

bool parse_bool(const char * value, bool default_value, bool & out) {
    if (!value || value[0] == '\0') {
        out = default_value;
        return true;
    }
    if (std::strcmp(value, "0") == 0) {
        out = false;
        return true;
    }
    if (std::strcmp(value, "1") == 0) {
        out = true;
        return true;
    }
    return false;
}

bool backend_is_cann(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    const char * name = ggml_backend_name(backend);
    if (!name) {
        return false;
    }
    std::string normalized(name);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return normalized.find("cann") != std::string::npos;
}

} // namespace

bool tts_device_head_parse_config(
        const char * head,
        const char * device_sampler,
        const char * trace,
        tts_device_head_config & config,
        std::string & error) {
    config = {};
    error.clear();

    const std::string mode = head && head[0] ? head : "cpu";
    if (mode == "cpu") {
        config.mode = tts_head_mode::cpu;
    } else if (mode == "cann") {
        config.mode = tts_head_mode::cann;
    } else {
        error = "OMNI_TTS_HEAD must be cpu or cann";
        return false;
    }

    if (!parse_bool(device_sampler, config.mode == tts_head_mode::cann, config.device_sampler)) {
        error = "OMNI_TTS_DEVICE_SAMPLER must be 0 or 1";
        return false;
    }
    if (!parse_bool(trace, false, config.trace)) {
        error = "OMNI_TTS_TRACE must be 0 or 1";
        return false;
    }
    if (config.mode == tts_head_mode::cann && !config.device_sampler) {
        error = "OMNI_TTS_HEAD=cann requires OMNI_TTS_DEVICE_SAMPLER=1";
        return false;
    }
    if (config.mode == tts_head_mode::cpu && config.device_sampler) {
        error = "OMNI_TTS_DEVICE_SAMPLER=1 requires OMNI_TTS_HEAD=cann";
        return false;
    }
    return true;
}

struct tts_device_head::impl {
    ggml_backend_t backend = nullptr; // borrowed

    ggml_context_ptr weight_ctx;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_context_ptr compute_ctx;
    ggml_gallocr_t allocator = nullptr;

    ggml_tensor * head_weight = nullptr;
    ggml_tensor * emb_weight = nullptr;
    ggml_tensor * hidden_input = nullptr;
    ggml_tensor * recent_ids = nullptr;
    ggml_tensor * penalty_neg = nullptr;
    ggml_tensor * penalty_pos = nullptr;
    ggml_tensor * eos_bias = nullptr;
    ggml_tensor * eos_id = nullptr;
    ggml_tensor * sampled = nullptr;
    ggml_tensor * embedding = nullptr;
    ggml_cgraph * graph = nullptr;

    llama_sampler * sampler = nullptr;

    int32_t hidden_size = 0;
    int32_t vocab_size = 0;
    int32_t top_k = 0;
    int32_t min_keep = 0;
    int32_t repetition_window = 0;
    float repetition_penalty = 1.0f;
    bool greedy = false;
    bool trace = false;

    ~impl() {
        if (sampler) {
            llama_sampler_free(sampler);
        }
        if (allocator) {
            ggml_gallocr_free(allocator);
        }
    }
};

tts_device_head::tts_device_head() : pimpl(new impl()) {}
tts_device_head::~tts_device_head() = default;

bool tts_device_head::initialize(
        const llama_device_tensor & hidden,
        const float * head_code,
        const float * emb_code,
        int32_t hidden_size,
        int32_t vocab_size,
        float temperature,
        float top_p,
        int32_t top_k,
        int32_t min_keep,
        float repetition_penalty,
        int32_t repetition_window,
        uint32_t seed,
        bool greedy,
        bool require_cann,
        bool trace,
        std::string & error) {
    error.clear();
    if (initialized()) {
        return true;
    }
    if (!hidden.tensor || !hidden.backend || !head_code || !emb_code) {
        error = "missing device hidden/backend or TTS weights";
        return false;
    }
    if (hidden_size <= 0 || vocab_size <= 1 || repetition_window <= 0 ||
        repetition_window > vocab_size || top_k < 0 || min_keep < 1 ||
        hidden.ne[0] != hidden_size || hidden.ne[1] != 1 ||
        hidden.type != GGML_TYPE_F32) {
        error = "invalid TTS head dimensions/sampler contract (requires F32 M=1)";
        return false;
    }

    auto * backend = static_cast<ggml_backend_t>(hidden.backend);
    if (require_cann && !backend_is_cann(backend)) {
        error = std::string("TTS device head requires CANN backend, got ") +
                (ggml_backend_name(backend) ? ggml_backend_name(backend) : "<unknown>");
        return false;
    }

    pimpl->backend = backend;
    pimpl->hidden_size = hidden_size;
    pimpl->vocab_size = vocab_size;
    pimpl->top_k = top_k;
    pimpl->min_keep = min_keep;
    pimpl->repetition_penalty = repetition_penalty;
    pimpl->repetition_window = repetition_window;
    pimpl->greedy = greedy;
    pimpl->trace = trace;

    const size_t weight_meta = 2 * ggml_tensor_overhead();
    pimpl->weight_ctx.reset(ggml_init({
        /*.mem_size   =*/ weight_meta,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    }));
    if (!pimpl->weight_ctx) {
        error = "failed to create TTS head weight context";
        reset();
        return false;
    }
    pimpl->head_weight = ggml_new_tensor_2d(
            pimpl->weight_ctx.get(), GGML_TYPE_F32, hidden_size, vocab_size);
    pimpl->emb_weight = ggml_new_tensor_2d(
            pimpl->weight_ctx.get(), GGML_TYPE_F32, hidden_size, vocab_size);
    ggml_set_name(pimpl->head_weight, "tts_head_weight");
    ggml_set_name(pimpl->emb_weight, "tts_emb_code_weight");

    pimpl->weight_buffer.reset(
            ggml_backend_alloc_ctx_tensors(pimpl->weight_ctx.get(), backend));
    if (!pimpl->weight_buffer) {
        error = "failed to allocate TTS head weights on backend";
        reset();
        return false;
    }
    const size_t weight_bytes = static_cast<size_t>(hidden_size) *
            static_cast<size_t>(vocab_size) * sizeof(float);
    ggml_backend_tensor_set(pimpl->head_weight, head_code, 0, weight_bytes);
    ggml_backend_tensor_set(pimpl->emb_weight, emb_code, 0, weight_bytes);

    pimpl->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (greedy) {
        llama_sampler_chain_add(pimpl->sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(pimpl->sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(pimpl->sampler, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(pimpl->sampler, llama_sampler_init_top_p(top_p, min_keep));
        llama_sampler_chain_add(pimpl->sampler, llama_sampler_init_dist(seed));
    }

    auto * buft = ggml_backend_get_default_buffer_type(backend);
    if (!pimpl->sampler->iface->backend_init ||
        !pimpl->sampler->iface->backend_init(pimpl->sampler, buft)) {
        error = "CANN backend does not support the requested TTS sampler chain";
        reset();
        return false;
    }

    constexpr int graph_nodes = 256;
    const size_t compute_meta =
            graph_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_nodes, false);
    pimpl->compute_ctx.reset(ggml_init({
        /*.mem_size   =*/ compute_meta,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    }));
    if (!pimpl->compute_ctx) {
        error = "failed to create TTS head compute context";
        reset();
        return false;
    }

    ggml_context * ctx = pimpl->compute_ctx.get();
    pimpl->hidden_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_set_input(pimpl->hidden_input);

    ggml_tensor * logits = ggml_mul_mat(ctx, pimpl->head_weight, pimpl->hidden_input);
    logits = ggml_reshape_1d(ctx, logits, vocab_size);

    if (!greedy) {
        pimpl->recent_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, repetition_window);
        pimpl->penalty_neg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, repetition_window);
        pimpl->penalty_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, repetition_window);
        pimpl->eos_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        pimpl->eos_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        for (auto * input : {
                 pimpl->recent_ids, pimpl->penalty_neg,
                 pimpl->penalty_pos, pimpl->eos_bias, pimpl->eos_id}) {
            ggml_set_input(input);
        }

        ggml_tensor * logits_2d = ggml_reshape_2d(ctx, logits, 1, vocab_size);
        ggml_tensor * recent = ggml_get_rows(ctx, logits_2d, pimpl->recent_ids);
        recent = ggml_reshape_1d(ctx, recent, repetition_window);
        ggml_tensor * sign = ggml_step(ctx, recent);
        ggml_tensor * scale_delta = ggml_sub(ctx, pimpl->penalty_pos, pimpl->penalty_neg);
        ggml_tensor * scales = ggml_add(
                ctx, pimpl->penalty_neg, ggml_mul(ctx, sign, scale_delta));
        ggml_tensor * adjusted = ggml_mul(ctx, recent, scales);
        adjusted = ggml_reshape_2d(ctx, adjusted, 1, repetition_window);
        logits_2d = ggml_set_rows(ctx, logits_2d, adjusted, pimpl->recent_ids);
        logits = ggml_reshape_1d(ctx, logits_2d, vocab_size);

        ggml_tensor * bias = ggml_fill(ctx, logits, 0.0f);
        bias = ggml_reshape_2d(ctx, bias, 1, vocab_size);
        bias = ggml_set_rows(
                ctx, bias, ggml_reshape_2d(ctx, pimpl->eos_bias, 1, 1), pimpl->eos_id);
        logits = ggml_add(ctx, logits, ggml_reshape_1d(ctx, bias, vocab_size));
    }

    pimpl->graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    llama_sampler_data sampler_data = {
        /*.logits      =*/ logits,
        /*.probs       =*/ nullptr,
        /*.sampled     =*/ nullptr,
        /*.candidates  =*/ nullptr,
    };
    pimpl->sampler->iface->backend_apply(
            pimpl->sampler, ctx, pimpl->graph, &sampler_data);
    pimpl->sampled = sampler_data.sampled;
    if (!pimpl->sampled || pimpl->sampled->type != GGML_TYPE_I32) {
        error = "TTS sampler did not produce an I32 device token";
        reset();
        return false;
    }
    pimpl->embedding = ggml_get_rows(ctx, pimpl->emb_weight, pimpl->sampled);
    ggml_set_name(pimpl->sampled, "tts_sampled_relative_token");
    ggml_set_name(pimpl->embedding, "tts_sampled_embedding");
    ggml_set_output(pimpl->sampled);
    ggml_set_output(pimpl->embedding);
    ggml_build_forward_expand(pimpl->graph, pimpl->sampled);
    ggml_build_forward_expand(pimpl->graph, pimpl->embedding);

    pimpl->allocator = ggml_gallocr_new(buft);
    if (!pimpl->allocator || !ggml_gallocr_alloc_graph(pimpl->allocator, pimpl->graph)) {
        error = "failed to allocate persistent TTS device head graph";
        reset();
        return false;
    }

    if (!greedy) {
        const int32_t eos = vocab_size - 1;
        ggml_backend_tensor_set(pimpl->eos_id, &eos, 0, sizeof(eos));
    }
    LOG_INF("TTS device head: initialized backend=%s hidden=%d vocab=%d sampler=%s "
            "stochastic_exactness=%s\n",
            ggml_backend_name(backend), hidden_size, vocab_size,
            greedy ? "argmax" : "temp+top-k+top-p+dist",
            greedy ? "exact" : "unverified");
    return true;
}

bool tts_device_head::forward(
        const llama_device_tensor & hidden,
        const tts_device_head_step & step,
        int32_t & selected_relative_token,
        llama_device_tensor & selected_embedding,
        std::string & error) {
    error.clear();
    selected_relative_token = -1;
    selected_embedding = {};
    if (!initialized()) {
        error = "TTS device head is not initialized";
        return false;
    }
    auto * source = static_cast<ggml_tensor *>(hidden.tensor);
    auto * source_backend = static_cast<ggml_backend_t>(hidden.backend);
    if (!source || source_backend != pimpl->backend ||
        hidden.type != GGML_TYPE_F32 || hidden.ne[0] != pimpl->hidden_size ||
        hidden.ne[1] != 1) {
        error = "TTS device hidden backend/type/shape changed";
        return false;
    }

    std::map<int32_t, int32_t> frequencies;
    if (!step.skip_repetition) {
        const size_t begin = step.recent_relative_tokens.size() >
                static_cast<size_t>(pimpl->repetition_window)
                ? step.recent_relative_tokens.size() - pimpl->repetition_window
                : 0;
        for (size_t i = begin; i < step.recent_relative_tokens.size(); ++i) {
            const int32_t token = step.recent_relative_tokens[i];
            if (token >= 0 && token < pimpl->vocab_size) {
                frequencies[token]++;
            }
        }
    }

    std::vector<int32_t> ids;
    std::vector<float> neg;
    std::vector<float> pos;
    ids.reserve(pimpl->repetition_window);
    neg.reserve(pimpl->repetition_window);
    pos.reserve(pimpl->repetition_window);
    std::set<int32_t> used;
    for (const auto & [token, frequency] : frequencies) {
        if (ids.size() >= static_cast<size_t>(pimpl->repetition_window)) {
            break;
        }
        const float alpha = std::pow(pimpl->repetition_penalty, frequency);
        ids.push_back(token);
        neg.push_back(alpha);
        pos.push_back(1.0f / alpha);
        used.insert(token);
    }
    for (int32_t token = 0;
         ids.size() < static_cast<size_t>(pimpl->repetition_window) &&
         token < pimpl->vocab_size;
         ++token) {
        if (used.insert(token).second) {
            ids.push_back(token);
            neg.push_back(1.0f);
            pos.push_back(1.0f);
        }
    }

    const float eos_bias = step.force_no_eos
            ? -std::numeric_limits<float>::infinity() : 0.0f;

    ggml_backend_tensor_copy(source, pimpl->hidden_input);
    if (!pimpl->greedy) {
        ggml_backend_tensor_set(pimpl->recent_ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
        ggml_backend_tensor_set(pimpl->penalty_neg, neg.data(), 0, neg.size() * sizeof(neg[0]));
        ggml_backend_tensor_set(pimpl->penalty_pos, pos.data(), 0, pos.size() * sizeof(pos[0]));
        ggml_backend_tensor_set(pimpl->eos_bias, &eos_bias, 0, sizeof(eos_bias));
    }

    if (pimpl->sampler->iface->backend_set_input) {
        pimpl->sampler->iface->backend_set_input(pimpl->sampler);
    }

    const auto start = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_graph_compute_async(pimpl->backend, pimpl->graph);
    if (status != GGML_STATUS_SUCCESS) {
        error = "TTS device head graph compute failed";
        return false;
    }
    ggml_backend_tensor_get_async(
            pimpl->backend, pimpl->sampled,
            &selected_relative_token, 0, sizeof(selected_relative_token));
    ggml_backend_synchronize(pimpl->backend);
    const auto stop = std::chrono::steady_clock::now();

    if (selected_relative_token < 0 || selected_relative_token >= pimpl->vocab_size) {
        error = "TTS device sampler produced out-of-range token";
        return false;
    }

    selected_embedding.tensor = pimpl->embedding;
    selected_embedding.backend = pimpl->backend;
    selected_embedding.type = static_cast<int32_t>(pimpl->embedding->type);
    for (int d = 0; d < 4; ++d) {
        selected_embedding.ne[d] = pimpl->embedding->ne[d];
    }
    if (pimpl->trace) {
        const double elapsed_us = std::chrono::duration<double, std::micro>(
                stop - start).count();
        LOG_INF("TTS device head step: token=%d eos=%d compute_sync_us=%.3f "
                "d2h_bytes=%zu hidden_d2h_bytes=0 logits_d2h_bytes=0 emb_h2d_bytes=0\n",
                selected_relative_token,
                selected_relative_token == pimpl->vocab_size - 1,
                elapsed_us, sizeof(selected_relative_token));
    }
    return true;
}

void tts_device_head::reset() {
    pimpl.reset(new impl());
}

bool tts_device_head::initialized() const {
    return pimpl && pimpl->backend && pimpl->graph && pimpl->allocator;
}

} // namespace omni
