// Wake word engine: audio -> features (TFLite Micro "microfrontend") -> streaming microWakeWord model
// -> probability -> sliding-window average -> detection.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float probability_cutoff;   // 0..1, detection when the averaged probability is above this
    int sliding_window;         // number of model outputs averaged (microWakeWord default: 5)
    int cooldown_ms;            // below-threshold audio needed before another detection
    size_t tensor_arena_bytes;  // first arena size to try (grown automatically if too small)
} ww_config_t;

typedef struct {
    int input_frames;           // feature frames per inference (the model's stride, normally 3)
    int feature_size;           // features per frame (40)
    float input_scale;
    int input_zero_point;
    size_t arena_size;          // allocated tensor arena
    size_t arena_used;          // bytes the model actually uses
    size_t heap_used;           // total heap the engine took (arena + frontend + variables)
} ww_info_t;

typedef struct {
    uint32_t inferences;        // model runs since the last call
    uint64_t busy_us;           // time spent in features + inference
    uint64_t invoke_us;         // of which inference (the rest is the feature frontend)
    uint32_t skipped;           // inferences skipped because it was quiet (the quiet-room gate)
    float max_avg_probability;  // highest averaged probability seen
    float last_avg_probability;
} ww_stats_t;

// Loads and checks the model. Prints a clear error if it is not a streaming microWakeWord model.
esp_err_t ww_init(const uint8_t *model, size_t model_len, const ww_config_t *cfg, ww_info_t *info);

// Feeds 16 kHz 16-bit mono samples. Returns true if the wake word was detected in this chunk. The model pauses
// after 1.6 s of quiet (the features keep running) and catches up on the last 300 ms when sound returns.
// *avg_probability (optional) receives the latest averaged probability.
bool ww_process(const int16_t *samples, size_t num_samples, float *avg_probability);

// Clears all model/feature state (used after the self-test). Detection is blocked for the first second again.
void ww_reset(void);

void ww_take_stats(ww_stats_t *out);

// Peak averaged probability and length of the most recent score event (sent to the server with each utterance)
void ww_last_event(float *peak, int *ms);

#ifdef __cplusplus
}
#endif
