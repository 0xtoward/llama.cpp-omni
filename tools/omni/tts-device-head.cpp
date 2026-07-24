#include "tts-device-head.h"
#include "e2e-trace.h"

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
    ggml_context_ptr input_ctx;
    ggml_backend_buffer_ptr input_buffer;
    ggml_context_ptr compute_ctx;
    ggml_gallocr_t allocator = nullptr;

    ggml_tensor * head_weight = nullptr;
    ggml_tensor * emb_weight = nullptr;
    ggml_tensor * hidden_input = nullptr;
    ggml_tensor * head_logits = nullptr;
    ggml_tensor * recent_ids = nullptr;
    ggml_tensor * penalty_neg = nullptr;
    ggml_tensor * penalty_pos = nullptr;
    ggml_tensor * eos_bias = nullptr;
    ggml_tensor * eos_id = nullptr;
    ggml_tensor * uniform = nullptr;
    ggml_tensor * top_p_floor_bias = nullptr;
    ggml_tensor * sample_rank_bias = nullptr;
    ggml_tensor * sample_probs = nullptr;
    ggml_tensor * sample_cdf = nullptr;
    ggml_tensor * sample_mask = nullptr;
    ggml_tensor * sample_scores = nullptr;
    ggml_tensor * candidate_order = nullptr;
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
    bool apply_top_k_p = false;
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
        bool greedy,
        bool apply_top_k_p,
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
        (apply_top_k_p &&
         (top_k < min_keep || top_k > vocab_size || top_p <= 0.0f || top_p > 1.0f)) ||
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
    pimpl->apply_top_k_p = apply_top_k_p;
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
    pimpl->input_ctx.reset(ggml_init({
        /*.mem_size   =*/ 16 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    }));
    if (!pimpl->input_ctx) {
        error = "failed to create TTS head persistent input context";
        reset();
        return false;
    }
    ggml_context * input_ctx = pimpl->input_ctx.get();
    pimpl->hidden_input = ggml_new_tensor_2d(
            input_ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_set_input(pimpl->hidden_input);

    ggml_tensor * logits = ggml_mul_mat(ctx, pimpl->head_weight, pimpl->hidden_input);
    pimpl->head_logits = logits;
    logits = ggml_reshape_1d(ctx, logits, vocab_size);

    if (!greedy) {
        pimpl->recent_ids = ggml_new_tensor_1d(
                input_ctx, GGML_TYPE_I32, repetition_window);
        pimpl->penalty_neg = ggml_new_tensor_1d(
                input_ctx, GGML_TYPE_F32, repetition_window);
        pimpl->penalty_pos = ggml_new_tensor_1d(
                input_ctx, GGML_TYPE_F32, repetition_window);
        pimpl->eos_bias = ggml_new_tensor_1d(
                input_ctx, GGML_TYPE_F32, 1);
        pimpl->eos_id = ggml_new_tensor_1d(
                input_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(pimpl->recent_ids, "tts_recent_ids");
        ggml_set_name(pimpl->penalty_neg, "tts_penalty_neg");
        ggml_set_name(pimpl->penalty_pos, "tts_penalty_pos");
        ggml_set_name(pimpl->eos_bias, "tts_eos_bias");
        ggml_set_name(pimpl->eos_id, "tts_eos_id");
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
    if (!greedy) {
        if (apply_top_k_p) {
            // Match nucleus_sampling_with_min_keep_tts exactly:
            // probabilities are normalized over the full vocabulary first,
            // then the top-k prefix is selected and top-p is evaluated without
            // renormalizing that prefix.
            ggml_tensor * full_probs = ggml_soft_max(ctx, sampler_data.logits);
            ggml_tensor * logits_2d = ggml_reshape_2d(
                    ctx, sampler_data.logits, 1, vocab_size);
            ggml_tensor * probs_2d = ggml_reshape_2d(
                    ctx, full_probs, 1, vocab_size);
            // Select a compact, sorted candidate set on device. The CANN
            // backend maps this to aclnnTopk, avoiding a full-vocabulary
            // ARGSORT and the fragile lifetime of its 6562-element index
            // buffer.
            ggml_tensor * top_k_indices =
                    ggml_top_k(ctx, sampler_data.logits, top_k);
            ggml_tensor * top_k_logits = ggml_reshape_1d(
                    ctx, ggml_get_rows(
                            ctx, logits_2d, top_k_indices), top_k);
            // GGML_OP_TOP_K intentionally does not guarantee rank order
            // across backends. Sort only the compact K-vector, then map the
            // local ranks back to vocabulary ids.
            ggml_tensor * compact_order =
                    ggml_argsort(
                            ctx, top_k_logits, GGML_SORT_ORDER_DESC);
            ggml_tensor * top_k_indices_2d =
                    ggml_reshape_2d(ctx, top_k_indices, 1, top_k);
            pimpl->candidate_order = ggml_reshape_1d(
                    ctx, ggml_get_rows(
                            ctx, top_k_indices_2d, compact_order), top_k);
            ggml_set_output(pimpl->candidate_order);
            ggml_tensor * sorted_logits = ggml_reshape_1d(
                    ctx, ggml_get_rows(
                            ctx, logits_2d, pimpl->candidate_order), top_k);
            ggml_tensor * sorted_probs = ggml_reshape_1d(
                    ctx, ggml_get_rows(
                            ctx, probs_2d, pimpl->candidate_order), top_k);

            // CPU keeps candidate i when it belongs to min_keep or when the
            // cumulative probability *before* i is still below top_p.
            // Express that directly instead of a dynamic set_rows at the
            // crossing index; the latter is both unnecessary and poorly
            // supported by device backends.
            ggml_tensor * cdf_before =
                    ggml_sub(ctx, ggml_cumsum(ctx, sorted_probs), sorted_probs);
            ggml_tensor * cdf_scaled =
                    ggml_scale(ctx, cdf_before, -1.0f);
            pimpl->top_p_floor_bias =
                    ggml_new_tensor_1d(
                            input_ctx, GGML_TYPE_F32, top_k);
            ggml_set_name(pimpl->top_p_floor_bias, "tts_top_p_floor_bias");
            ggml_set_input(pimpl->top_p_floor_bias);
            cdf_scaled = ggml_add(ctx, cdf_scaled, pimpl->top_p_floor_bias);
            ggml_tensor * keep_mask = ggml_step(ctx, cdf_scaled);

            sampler_data.logits =
                    ggml_add(ctx, sorted_logits, ggml_log(ctx, keep_mask));
            sampler_data.candidates = pimpl->candidate_order;
        }

        // Reproduce the legacy CPU sampler with a caller-provided float
        // uniform. The large probability/logit tensors never leave device.
        pimpl->uniform = ggml_new_tensor_1d(
                input_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(pimpl->uniform, "tts_sampling_uniform");
        ggml_set_input(pimpl->uniform);

        ggml_tensor * probs = ggml_soft_max(ctx, sampler_data.logits);
        ggml_tensor * cdf = ggml_cumsum(ctx, probs);
        ggml_tensor * diff = ggml_sub(ctx, cdf, pimpl->uniform);
        ggml_tensor * mask = ggml_step(ctx, diff);
        pimpl->sample_probs = probs;
        pimpl->sample_cdf = cdf;
        pimpl->sample_mask = mask;
        // mask is [0, ..., 0, 1, ..., 1]. CANN ARGMAX does not promise the
        // first index on ties, so multiply by a fixed descending rank. The
        // first CDF entry crossing the uniform becomes the unique maximum.
        // This also avoids the unsupported F32 -> I32 CPY/cast path.
        const int64_t n_candidates = ggml_nelements(mask);
        pimpl->sample_rank_bias =
                ggml_new_tensor_1d(
                        input_ctx, GGML_TYPE_F32, n_candidates);
        ggml_set_name(pimpl->sample_rank_bias, "tts_sample_rank_bias");
        ggml_set_input(pimpl->sample_rank_bias);
        pimpl->sample_scores = ggml_mul(
                ctx, mask, pimpl->sample_rank_bias);
        // Keep the complete sampling chain alive until the sampled rank and
        // vocabulary-id gather have both finished.  On CANN, allowing
        // gallocr to recycle these small intermediates early can overwrite
        // the top-k/top-p mask before the final gather.  Diagnostic dumps
        // used to hide this by marking the same tensors as outputs.
        ggml_set_output(pimpl->sample_probs);
        ggml_set_output(pimpl->sample_cdf);
        ggml_set_output(pimpl->sample_mask);
        ggml_set_output(pimpl->sample_scores);
        ggml_tensor * selected_index =
                ggml_argmax(ctx, pimpl->sample_scores);
        if (std::getenv("OMNI_TTS_DEBUG_DUMP")) {
            if (pimpl->candidate_order) {
                ggml_set_output(pimpl->candidate_order);
            }
        }

        sampler_data.sampled = selected_index;
        if (sampler_data.candidates) {
            ggml_tensor * candidates = ggml_reshape_2d(
                    ctx, sampler_data.candidates, 1,
                    ggml_nelements(sampler_data.candidates));
            sampler_data.sampled = ggml_get_rows(ctx, candidates, selected_index);
        }
        ggml_set_name(sampler_data.sampled, "tts_fixed_uniform_sample");
    }
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

    for (int i = 0; i < ggml_graph_n_nodes(pimpl->graph); ++i) {
        ggml_tensor * node = ggml_graph_node(pimpl->graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            error = std::string("backend ") + ggml_backend_name(backend) +
                    " lacks TTS sampler op " + ggml_op_name(node->op);
            reset();
            return false;
        }
    }

    pimpl->input_buffer.reset(
            ggml_backend_alloc_ctx_tensors(input_ctx, backend));
    if (!pimpl->input_buffer) {
        error = "failed to allocate persistent TTS device-head inputs";
        reset();
        return false;
    }
    pimpl->allocator = ggml_gallocr_new(buft);
    if (!pimpl->allocator || !ggml_gallocr_alloc_graph(pimpl->allocator, pimpl->graph)) {
        error = "failed to allocate persistent TTS device head graph";
        reset();
        return false;
    }

    if (!greedy) {
        const int32_t eos = vocab_size - 1;
        ggml_backend_tensor_set(pimpl->eos_id, &eos, 0, sizeof(eos));
        const int64_t n_candidates = apply_top_k_p ? top_k : vocab_size;
        std::vector<float> rank_bias(n_candidates);
        for (int64_t i = 0; i < n_candidates; ++i) {
            rank_bias[i] = static_cast<float>(n_candidates - i);
        }
        ggml_backend_tensor_set(
                pimpl->sample_rank_bias,
                rank_bias.data(), 0,
                rank_bias.size() * sizeof(rank_bias[0]));
        if (apply_top_k_p) {
            std::vector<float> floor_bias(top_k, top_p);
            for (int32_t i = 0; i < min_keep; ++i) {
                floor_bias[i] = std::numeric_limits<float>::infinity();
            }
            ggml_backend_tensor_set(
                    pimpl->top_p_floor_bias,
                    floor_bias.data(), 0,
                    floor_bias.size() * sizeof(floor_bias[0]));
        }
    }
    LOG_INF("TTS device head: initialized backend=%s hidden=%d vocab=%d sampler=%s "
            "stochastic_exactness=%s\n",
            ggml_backend_name(backend), hidden_size, vocab_size,
            greedy ? "argmax" :
                (apply_top_k_p ? "temp+top-k+top-p+fixed-uniform" :
                                 "temp+fixed-uniform"),
            "exact_replay");
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
    if (!pimpl->greedy &&
        (!step.has_uniform || !std::isfinite(step.uniform) ||
         step.uniform < 0.0f || step.uniform >= 1.0f)) {
        error = "stochastic TTS device head requires a finite uniform in [0,1)";
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

    {
        e2e_trace::span hidden_bridge_trace(
            "tts", "hidden_bridge_d2d", ggml_backend_name(pimpl->backend));
        ggml_backend_tensor_copy(source, pimpl->hidden_input);
    }
    if (!pimpl->greedy) {
        ggml_backend_tensor_set(pimpl->recent_ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
        ggml_backend_tensor_set(pimpl->penalty_neg, neg.data(), 0, neg.size() * sizeof(neg[0]));
        ggml_backend_tensor_set(pimpl->penalty_pos, pos.data(), 0, pos.size() * sizeof(pos[0]));
        ggml_backend_tensor_set(pimpl->eos_bias, &eos_bias, 0, sizeof(eos_bias));
        ggml_backend_tensor_set(pimpl->uniform, &step.uniform, 0, sizeof(step.uniform));
    }

    const auto start = std::chrono::steady_clock::now();
    {
        e2e_trace::span device_head_trace(
            "tts", "device_head_graph", ggml_backend_name(pimpl->backend));
        const ggml_status status =
            ggml_backend_graph_compute_async(pimpl->backend, pimpl->graph);
        if (status != GGML_STATUS_SUCCESS) {
            error = "TTS device head graph compute failed";
            return false;
        }
        ggml_backend_tensor_get_async(
                pimpl->backend, pimpl->sampled,
                &selected_relative_token, 0, sizeof(selected_relative_token));
        {
            e2e_trace::span scalar_sync_trace(
                "tts", "token_scalar_sync", ggml_backend_name(pimpl->backend));
            ggml_backend_synchronize(pimpl->backend);
        }
    }
    const auto stop = std::chrono::steady_clock::now();

    if (selected_relative_token < 0 || selected_relative_token >= pimpl->vocab_size) {
        error = "TTS device sampler produced out-of-range token";
        return false;
    }
    if (std::getenv("OMNI_TTS_DEBUG_DUMP") && !pimpl->greedy) {
        const int64_t n = ggml_nelements(pimpl->sample_probs);
        std::vector<float> probs(n);
        std::vector<float> cdf(n);
        std::vector<float> mask(n);
        std::vector<float> scores(n);
        ggml_backend_tensor_get(
                pimpl->sample_probs, probs.data(), 0,
                probs.size() * sizeof(probs[0]));
        ggml_backend_tensor_get(
                pimpl->sample_cdf, cdf.data(), 0,
                cdf.size() * sizeof(cdf[0]));
        ggml_backend_tensor_get(
                pimpl->sample_mask, mask.data(), 0,
                mask.size() * sizeof(mask[0]));
        ggml_backend_tensor_get(
                pimpl->sample_scores, scores.data(), 0,
                scores.size() * sizeof(scores[0]));
        std::fprintf(
                stderr,
                "TTS_DEVICE_DEBUG uniform=%.9g selected=%d n=%lld\n",
                step.uniform,
                selected_relative_token,
                static_cast<long long>(n));
        for (int64_t i = 0; i < std::min<int64_t>(n, 32); ++i) {
            std::fprintf(
                    stderr,
                    "TTS_DEVICE_DEBUG i=%lld p=%.9g cdf=%.9g mask=%.9g "
                    "score=%.9g\n",
                    static_cast<long long>(i),
                    probs[i],
                    cdf[i],
                    mask[i],
                    scores[i]);
        }
        if (pimpl->candidate_order) {
            std::vector<int32_t> order(
                    ggml_nelements(pimpl->candidate_order));
            ggml_backend_tensor_get(
                    pimpl->candidate_order,
                    order.data(), 0,
                    order.size() * sizeof(order[0]));
            std::fprintf(stderr, "TTS_DEVICE_DEBUG order=");
            for (int64_t i = 0; i < std::min<int64_t>(
                     static_cast<int64_t>(order.size()), 32); ++i) {
                std::fprintf(
                        stderr, "%s%d", i ? "," : "", order[i]);
            }
            std::fprintf(stderr, "\n");
        }
    }

    selected_embedding.tensor = pimpl->embedding;
    selected_embedding.backend = pimpl->backend;
    selected_embedding.type = static_cast<int32_t>(pimpl->embedding->type);
    for (int d = 0; d < 4; ++d) {
        selected_embedding.ne[d] = pimpl->embedding->ne[d];
    }
    auto & trace_state = e2e_trace::global_state();
    if (trace_state.tensor_enabled()) {
        const auto ids = e2e_trace::current_context();
        auto record = [&](const char * action,
                          const char * role,
                          const char * producer,
                          const char * consumer,
                          const char * shape,
                          const char * transfer,
                          const char * claim,
                          ggml_tensor * tensor,
                          uint64_t bytes) {
            e2e_trace::tensor_record row;
            row.ids = ids;
            row.stage = "tts";
            row.action = action;
            row.tensor_role = role;
            row.producer = producer;
            row.consumer = consumer;
            row.shape = shape;
            row.dtype = "f32";
            row.device = ggml_backend_name(pimpl->backend);
            row.transfer = transfer;
            row.residency_claim = claim;
            row.bytes = bytes;
            row.buffer_id = reinterpret_cast<uintptr_t>(
                tensor ? tensor->data : nullptr);
            row.reused = true;
            trace_state.append_tensor(row);
        };
        record(
            "hidden_bridge",
            "tts_hidden",
            "minicpm_tts",
            "tts_code_head",
            "[1,768]",
            "device_to_device",
            "host_roundtrip_removed",
            pimpl->hidden_input,
            static_cast<uint64_t>(pimpl->hidden_size) * sizeof(float));
        record(
            "head_logits",
            "tts_code_logits",
            "tts_code_head",
            "device_sampler",
            "[1,6562]",
            "none",
            "host_roundtrip_removed",
            pimpl->head_logits,
            static_cast<uint64_t>(pimpl->vocab_size) * sizeof(float));
        record(
            "embedding_gather",
            "tts_next_embedding",
            "device_gather",
            "minicpm_tts_decode",
            "[1,768]",
            "device_to_device",
            "host_roundtrip_removed",
            pimpl->embedding,
            static_cast<uint64_t>(pimpl->hidden_size) * sizeof(float));
        record(
            "sampled_token",
            "tts_token_scalar",
            "device_sampler",
            "host_control",
            "[1]",
            "device_to_host",
            "required_control_scalar",
            pimpl->sampled,
            sizeof(selected_relative_token));
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
