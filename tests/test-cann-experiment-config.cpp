#include "../ggml/src/ggml-cann/experiment-config.h"

#include <cassert>
#include <iostream>

int main() {
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, true, true);
        assert(parsed);
        assert(parsed.config.graph == ggml_cann_graph_experiment::stock);
        assert(parsed.config.fusion == ggml_cann_fusion_experiment::add_rms);
        assert(parsed.config.layer_engine == ggml_cann_layer_engine::ggml);
        assert(!parsed.config.trace);
    }
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            "DECODE_BUCKET", "qk_rope_kv", "ascendc_ffn", "yes", false, false);
        assert(parsed);
        assert(parsed.config.graph == ggml_cann_graph_experiment::decode_bucket);
        assert(parsed.config.fusion == ggml_cann_fusion_experiment::qk_rope_kv);
        assert(parsed.config.layer_engine == ggml_cann_layer_engine::ascendc_ffn);
        assert(parsed.config.trace);
    }
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            "mystery", "none", "ggml", "0", false, false);
        assert(!parsed);
    }
    {
        const auto parsed = ggml_cann_parse_experiment_config(
            "off", "none", "ggml", "maybe", false, false);
        assert(!parsed);
    }

    std::cout << "CANN experiment config tests passed\n";
    return 0;
}
