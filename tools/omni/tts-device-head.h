#pragma once

#include "../../src/llama-ext.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace omni {

enum class tts_head_mode {
    cpu,
    cann,
};

struct tts_device_head_config {
    tts_head_mode mode = tts_head_mode::cpu;
    bool device_sampler = false;
    bool trace = false;
    bool suppress_model_logits = true;
};

bool tts_device_head_parse_config(
        const char * head,
        const char * device_sampler,
        const char * trace,
        const char * suppress_model_logits,
        tts_device_head_config & config,
        std::string & error);

struct tts_device_head_step {
    std::vector<int32_t> recent_relative_tokens;
    bool skip_repetition = false;
    bool force_no_eos = false;
    // Required for stochastic mode. Supplying the CPU-generated uniform makes
    // CPU and device sampling replayable without copying logits to the host.
    bool has_uniform = false;
    float uniform = 0.0f;
};

// Transfer ledger for one forward() call.  Initialization-time weight uploads
// are intentionally excluded.  The selected token also carries the stop/EOS
// decision, so a production step needs one int32 control scalar D2H and no
// separate stop-flag transfer.
struct tts_device_head_transfer_stats {
    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    uint64_t d2d_bytes = 0;
    uint64_t control_scalar_d2h_bytes = 0;
    uint64_t hidden_d2h_bytes = 0;
    uint64_t logits_d2h_bytes = 0;
    uint64_t embedding_h2d_bytes = 0;
    uint64_t embedding_d2h_bytes = 0;
    uint64_t diagnostic_d2h_bytes = 0;

    bool production_contract_ok() const {
        return d2h_bytes == sizeof(int32_t) &&
               control_scalar_d2h_bytes == sizeof(int32_t) &&
               hidden_d2h_bytes == 0 &&
               logits_d2h_bytes == 0 &&
               embedding_h2d_bytes == 0 &&
               embedding_d2h_bytes == 0 &&
               diagnostic_d2h_bytes == 0;
    }
};

// Persistent accelerator-side MiniCPMTTS code head.
//
// The runner owns its head/embedding weights and compute graph, but borrows the
// backend from the TTS llama_context. A successful forward copies only the
// selected int32 token to the host; logits and the selected embedding stay on
// the backend.
class tts_device_head {
public:
    tts_device_head();
    ~tts_device_head();

    tts_device_head(const tts_device_head &) = delete;
    tts_device_head & operator=(const tts_device_head &) = delete;

    bool initialize(
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
            std::string & error);

    bool forward(
            const llama_device_tensor & hidden,
            const tts_device_head_step & step,
            int32_t & selected_relative_token,
            llama_device_tensor & selected_embedding,
            std::string & error);

    void reset();
    bool initialized() const;
    tts_device_head_transfer_stats last_transfer_stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace omni
