#include "../ggml/src/ggml-cann/experiment-config.h"

#include <cassert>
#include <iostream>

int main() {
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true, true);
        assert(parsed);
        assert(parsed.config.graph == ggml_cann_graph_experiment::stock);
        assert(parsed.config.fusion == ggml_cann_fusion_experiment::add_rms);
        assert(parsed.config.layer_engine == ggml_cann_layer_engine::ggml);
        assert(!parsed.config.trace);
    }
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            "DECODE_BUCKET", std::nullopt, "qk_rope_kv", "ascendc_ffn", "yes", false, false);
        assert(parsed);
        assert(parsed.config.graph == ggml_cann_graph_experiment::decode_bucket);
        assert(parsed.config.fusion == ggml_cann_fusion_experiment::qk_rope_kv);
        assert(parsed.config.layer_engine == ggml_cann_layer_engine::ascendc_ffn);
        assert(parsed.config.trace);
    }
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            "mystery", std::nullopt, "none", "ggml", "0", false, false);
        assert(!parsed);
    }
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            "off", std::nullopt, "none", "ggml", "maybe", false, false);
        assert(!parsed);
    }
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            "stage_exact", "token2mel,tts_ar", "none", "ggml", "1", false, false);
        assert(parsed);
        assert(parsed.config.graph == ggml_cann_graph_experiment::stage_exact);
        assert(ggml_cann_graph_stage_enabled(parsed.config, ggml_cann_graph_stage::token2mel));
        assert(ggml_cann_graph_stage_enabled(parsed.config, ggml_cann_graph_stage::tts_ar));
        assert(ggml_cann_graph_stages_name(parsed.config) == "tts_ar,token2mel");
    }
    for (const char * invalid : {
             "", "token2mel,", ",token2mel", "token2mel,token2mel",
             "token2mel, tts_ar", "hift" }) {
        const auto parsed = ggml_cann_parse_experiment_config(
            "stage_exact", invalid, "none", "ggml", "0", false, false);
        assert(!parsed);
    }
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            "stock", "token2mel", "none", "ggml", "0", false, false);
        assert(!parsed);
    }
    // stage_exact initialization may capture; serving hot paths may only hit
    // an existing graph or fall back to eager.
    assert(ggml_cann_select_stage_exact_action(true, false, true) ==
           ggml_cann_stage_exact_action::capture);
    assert(ggml_cann_select_stage_exact_action(true, true, false) ==
           ggml_cann_stage_exact_action::hit);
    assert(ggml_cann_select_stage_exact_action(true, false, false) ==
           ggml_cann_stage_exact_action::eager);
    assert(ggml_cann_select_stage_exact_action(false, false, true) ==
           ggml_cann_stage_exact_action::eager);

    std::cout << "CANN experiment config tests passed\n";
    return 0;
}
