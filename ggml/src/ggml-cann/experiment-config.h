#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

enum class ggml_cann_graph_experiment {
    off,
    stock,
    decode_bucket,
    layer_islands,
    stage_exact,
};

enum class ggml_cann_graph_stage : uint32_t {
    tts_ar    = 1u << 0,
    token2mel = 1u << 1,
    hift      = 1u << 2,
};

enum class ggml_cann_stage_exact_action {
    eager,
    hit,
    capture,
};

enum class ggml_cann_fusion_experiment {
    none,
    add_rms,
    kv_pair,
    qk_rope_kv,
    swiglu,
    all,
};

enum class ggml_cann_layer_engine {
    ggml,
    atb,
    ascendc_ffn,
    ascendc_layer,
};

struct ggml_cann_experiment_config {
    ggml_cann_graph_experiment  graph       = ggml_cann_graph_experiment::off;
    uint32_t                    graph_stages = 0;
    ggml_cann_fusion_experiment fusion      = ggml_cann_fusion_experiment::none;
    ggml_cann_layer_engine      layer_engine = ggml_cann_layer_engine::ggml;
    bool                        trace       = false;
};

struct ggml_cann_experiment_parse_result {
    ggml_cann_experiment_config config;
    std::string                 error;

    explicit operator bool() const {
        return error.empty();
    }
};

inline std::string ggml_cann_experiment_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

inline const char * ggml_cann_experiment_name(ggml_cann_graph_experiment value) {
    switch (value) {
        case ggml_cann_graph_experiment::off:           return "off";
        case ggml_cann_graph_experiment::stock:         return "stock";
        case ggml_cann_graph_experiment::decode_bucket: return "decode_bucket";
        case ggml_cann_graph_experiment::layer_islands: return "layer_islands";
        case ggml_cann_graph_experiment::stage_exact:   return "stage_exact";
    }
    return "invalid";
}

inline bool ggml_cann_graph_stage_enabled(
        const ggml_cann_experiment_config & config,
        ggml_cann_graph_stage stage) {
    return (config.graph_stages & static_cast<uint32_t>(stage)) != 0;
}

inline std::string ggml_cann_graph_stages_name(const ggml_cann_experiment_config & config) {
    std::string result;
    if (ggml_cann_graph_stage_enabled(config, ggml_cann_graph_stage::tts_ar)) {
        result = "tts_ar";
    }
    if (ggml_cann_graph_stage_enabled(config, ggml_cann_graph_stage::token2mel)) {
        if (!result.empty()) {
            result += ",";
        }
        result += "token2mel";
    }
    if (ggml_cann_graph_stage_enabled(config, ggml_cann_graph_stage::hift)) {
        if (!result.empty()) {
            result += ",";
        }
        result += "hift";
    }
    return result;
}

inline ggml_cann_stage_exact_action ggml_cann_select_stage_exact_action(
        bool stage_enabled,
        bool cache_hit,
        bool allow_capture) {
    if (!stage_enabled) {
        return ggml_cann_stage_exact_action::eager;
    }
    if (cache_hit) {
        return ggml_cann_stage_exact_action::hit;
    }
    return allow_capture ? ggml_cann_stage_exact_action::capture
                         : ggml_cann_stage_exact_action::eager;
}

inline const char * ggml_cann_experiment_name(ggml_cann_fusion_experiment value) {
    switch (value) {
        case ggml_cann_fusion_experiment::none:       return "none";
        case ggml_cann_fusion_experiment::add_rms:    return "add_rms";
        case ggml_cann_fusion_experiment::kv_pair:    return "kv_pair";
        case ggml_cann_fusion_experiment::qk_rope_kv: return "qk_rope_kv";
        case ggml_cann_fusion_experiment::swiglu:     return "swiglu";
        case ggml_cann_fusion_experiment::all:        return "all";
    }
    return "invalid";
}

inline const char * ggml_cann_experiment_name(ggml_cann_layer_engine value) {
    switch (value) {
        case ggml_cann_layer_engine::ggml:          return "ggml";
        case ggml_cann_layer_engine::atb:           return "atb";
        case ggml_cann_layer_engine::ascendc_ffn:   return "ascendc_ffn";
        case ggml_cann_layer_engine::ascendc_layer: return "ascendc_layer";
    }
    return "invalid";
}

inline std::optional<bool> ggml_cann_experiment_parse_bool(const std::string & raw) {
    const std::string value = ggml_cann_experiment_lower(raw);
    if (value == "1" || value == "on" || value == "true" || value == "yes") {
        return true;
    }
    if (value == "0" || value == "off" || value == "false" || value == "no" || value.empty()) {
        return false;
    }
    return std::nullopt;
}

inline ggml_cann_experiment_parse_result ggml_cann_parse_experiment_config(
        const std::optional<std::string> & graph,
        const std::optional<std::string> & graph_stages,
        const std::optional<std::string> & fusion,
        const std::optional<std::string> & layer_engine,
        const std::optional<std::string> & trace,
        bool legacy_graph_enabled,
        bool legacy_add_rms_enabled) {
    ggml_cann_experiment_parse_result result;

    const std::string graph_value = ggml_cann_experiment_lower(
        graph.value_or(legacy_graph_enabled ? "stock" : "off"));
    if (graph_value == "off") {
        result.config.graph = ggml_cann_graph_experiment::off;
    } else if (graph_value == "stock") {
        result.config.graph = ggml_cann_graph_experiment::stock;
    } else if (graph_value == "decode_bucket") {
        result.config.graph = ggml_cann_graph_experiment::decode_bucket;
    } else if (graph_value == "layer_islands") {
        result.config.graph = ggml_cann_graph_experiment::layer_islands;
    } else if (graph_value == "stage_exact") {
        result.config.graph = ggml_cann_graph_experiment::stage_exact;
    } else {
        result.error = "GGML_CANN_GRAPH_EXPERIMENT must be off|stock|decode_bucket|layer_islands|stage_exact";
        return result;
    }

    const std::string stages_value = ggml_cann_experiment_lower(graph_stages.value_or(""));
    if (result.config.graph != ggml_cann_graph_experiment::stage_exact) {
        if (!stages_value.empty()) {
            result.error = "GGML_CANN_GRAPH_STAGES is only valid with GGML_CANN_GRAPH_EXPERIMENT=stage_exact";
            return result;
        }
    } else {
        if (stages_value.empty()) {
            result.error = "GGML_CANN_GRAPH_STAGES must list tts_ar, token2mel, and/or hift for stage_exact";
            return result;
        }
        if (std::any_of(stages_value.begin(), stages_value.end(), [](unsigned char ch) {
                return std::isspace(ch);
            })) {
            result.error = "GGML_CANN_GRAPH_STAGES must not contain whitespace";
            return result;
        }
        size_t begin = 0;
        while (begin <= stages_value.size()) {
            const size_t end = stages_value.find(',', begin);
            const std::string item = stages_value.substr(
                begin, end == std::string::npos ? std::string::npos : end - begin);
            uint32_t bit = 0;
            if (item == "tts_ar") {
                bit = static_cast<uint32_t>(ggml_cann_graph_stage::tts_ar);
            } else if (item == "token2mel") {
                bit = static_cast<uint32_t>(ggml_cann_graph_stage::token2mel);
            } else if (item == "hift") {
                bit = static_cast<uint32_t>(ggml_cann_graph_stage::hift);
            } else {
                result.error = "GGML_CANN_GRAPH_STAGES accepts only tts_ar,token2mel,hift";
                return result;
            }
            if ((result.config.graph_stages & bit) != 0) {
                result.error = "GGML_CANN_GRAPH_STAGES must not contain duplicates";
                return result;
            }
            result.config.graph_stages |= bit;
            if (end == std::string::npos) {
                break;
            }
            begin = end + 1;
        }
    }

    const std::string fusion_value = ggml_cann_experiment_lower(
        fusion.value_or(legacy_add_rms_enabled ? "add_rms" : "none"));
    if (fusion_value == "none") {
        result.config.fusion = ggml_cann_fusion_experiment::none;
    } else if (fusion_value == "add_rms") {
        result.config.fusion = ggml_cann_fusion_experiment::add_rms;
    } else if (fusion_value == "kv_pair") {
        result.config.fusion = ggml_cann_fusion_experiment::kv_pair;
    } else if (fusion_value == "qk_rope_kv") {
        result.config.fusion = ggml_cann_fusion_experiment::qk_rope_kv;
    } else if (fusion_value == "swiglu") {
        result.config.fusion = ggml_cann_fusion_experiment::swiglu;
    } else if (fusion_value == "all") {
        result.config.fusion = ggml_cann_fusion_experiment::all;
    } else {
        result.error = "GGML_CANN_FUSION_EXPERIMENT must be none|add_rms|kv_pair|qk_rope_kv|swiglu|all";
        return result;
    }

    const std::string engine_value = ggml_cann_experiment_lower(layer_engine.value_or("ggml"));
    if (engine_value == "ggml") {
        result.config.layer_engine = ggml_cann_layer_engine::ggml;
    } else if (engine_value == "atb") {
        result.config.layer_engine = ggml_cann_layer_engine::atb;
    } else if (engine_value == "ascendc_ffn") {
        result.config.layer_engine = ggml_cann_layer_engine::ascendc_ffn;
    } else if (engine_value == "ascendc_layer") {
        result.config.layer_engine = ggml_cann_layer_engine::ascendc_layer;
    } else {
        result.error = "GGML_CANN_LAYER_ENGINE must be ggml|atb|ascendc_ffn|ascendc_layer";
        return result;
    }

    const auto trace_value = ggml_cann_experiment_parse_bool(trace.value_or("0"));
    if (!trace_value.has_value()) {
        result.error = "GGML_CANN_EXPERIMENT_TRACE must be a strict boolean";
        return result;
    }
    result.config.trace = *trace_value;
    return result;
}
