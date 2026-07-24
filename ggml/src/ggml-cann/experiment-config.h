#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

enum class ggml_cann_graph_experiment {
    off,
    stock,
    decode_bucket,
    layer_islands,
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
    }
    return "invalid";
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
    } else {
        result.error = "GGML_CANN_GRAPH_EXPERIMENT must be off|stock|decode_bucket|layer_islands";
        return result;
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
