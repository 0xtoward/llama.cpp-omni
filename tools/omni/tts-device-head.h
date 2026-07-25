#pragma once

#include "../../src/llama-ext.h"

#include <array>
#include <cstdint>
#include <functional>
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
    enum llama_output_contract base_output = LLAMA_OUTPUT_DEFAULT;
    int32_t device_burst = 1;
};

bool tts_device_head_parse_config(
        const char * head,
        const char * device_sampler,
        const char * trace,
        const char * suppress_model_logits,
        const char * base_output,
        const char * device_burst,
        tts_device_head_config & config,
        std::string & error);

// Protocol state for the future device-resident four-code loop.
//
// This class deliberately contains no backend execution.  It defines and
// validates the state that a CANN implementation must keep on device:
// recent_ids, token/stop rings, valid_count and epoch.  Keeping the protocol
// independently testable lets the service fail closed until both the decoder
// loop and KV suffix rollback are wired to this exact contract.
struct tts_device_burst_event {
    uint64_t epoch = 0;
    std::array<int32_t, 4> tokens = {};
    std::array<uint8_t, 4> stops = {};
    int32_t valid_count = 0;
    int32_t accepted_count = 0;
    int32_t kv_rollback_count = 0;
};

class tts_device_burst_state {
public:
    static constexpr int32_t max_burst = 4;
    static constexpr int32_t recent_capacity = 16;

    explicit tts_device_burst_state(int32_t burst_width = 1);

    bool reset(uint64_t epoch, std::string & error);
    bool cancel(uint64_t next_epoch, std::string & error);
    bool begin(uint64_t epoch, std::string & error);
    bool append(
            uint64_t epoch,
            int32_t relative_token,
            bool stop,
            std::string & error);
    bool finish(
            uint64_t epoch,
            bool keep_stop_embedding,
            tts_device_burst_event & event,
            std::string & error);

    uint64_t epoch() const;
    int32_t burst_width() const;
    int32_t valid_count() const;
    std::vector<int32_t> recent_tokens() const;

private:
    int32_t width = 1;
    uint64_t current_epoch = 0;
    bool active = false;
    std::array<int32_t, recent_capacity> recent_ids = {};
    int32_t recent_begin = 0;
    int32_t recent_count = 0;
    std::array<int32_t, max_burst> token_ring = {};
    std::array<uint8_t, max_burst> stop_ring = {};
    int32_t pending_count = 0;
};

struct tts_device_head_step {
    std::vector<int32_t> recent_relative_tokens;
    bool skip_repetition = false;
    bool force_no_eos = false;
    // Required for stochastic mode. Supplying the CPU-generated uniform makes
    // CPU and device sampling replayable without copying logits to the host.
    bool has_uniform = false;
    float uniform = 0.0f;
    // Diagnostic identity only.  These values never participate in sampling.
    int32_t token_index = -1;
    int32_t n_past = -1;
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

    // Capability primitive for a true four-code device burst.  Uniforms are
    // uploaded once; four head graphs enqueue without token D2H; finish_burst
    // reads the compact token ring once.  Until exact device-side repetition
    // state exists, stochastic steps must set skip_repetition=true.
    bool begin_burst(
            uint64_t epoch,
            const std::array<float, 4> & uniforms,
            std::string & error);
    bool forward_burst_step(
            uint64_t epoch,
            const llama_device_tensor & hidden,
            const tts_device_head_step & step,
            llama_device_tensor & selected_embedding,
            std::string & error);
    bool finish_burst(
            uint64_t epoch,
            tts_device_burst_event & event,
            std::string & error);
    void cancel_burst(uint64_t next_epoch);

    void reset();
    bool initialized() const;
    tts_device_head_transfer_stats last_transfer_stats() const;

private:
    bool forward_impl(
            const llama_device_tensor & hidden,
            const tts_device_head_step & step,
            bool defer_token,
            int32_t burst_slot,
            int32_t & selected_relative_token,
            llama_device_tensor & selected_embedding,
            std::string & error);

    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Shared, unit-testable contract used by the service after a speculative
// burst.  remove_suffix must remove [new_n_past, old_n_past) from sequence 0.
bool tts_device_burst_rollback_kv(
        int32_t & n_past,
        int32_t rollback_count,
        const std::function<bool(int32_t, int32_t)> & remove_suffix,
        std::string & error);

} // namespace omni
