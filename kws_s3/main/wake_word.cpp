#include "wake_word.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"

#include "frontend.h"
#include "frontend_util.h"

#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "wake_word";

// main/fast_ops.cc: same results as TFLite Micro's kernels, the copy plan is worked out once
TFLMRegistration Register_STRIDED_SLICE_FAST();
TFLMRegistration Register_CONCATENATION_FAST();
TFLMRegistration Register_SPLIT_V_FAST();
#if CONFIG_KWS_FAST_SLICE
#define KWS_STRIDED_SLICE Register_STRIDED_SLICE_FAST()
#define KWS_CONCATENATION Register_CONCATENATION_FAST()
#define KWS_SPLIT_V Register_SPLIT_V_FAST()
#else
#define KWS_STRIDED_SLICE tflite::Register_STRIDED_SLICE()
#define KWS_CONCATENATION tflite::Register_CONCATENATION()
#define KWS_SPLIT_V tflite::Register_SPLIT_V()
#endif

#if CONFIG_KWS_PROFILE_OPS
#include "esp_cpu.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
extern "C" uint32_t g_frontend_prof_cycles[8];  // frontend.c (same Kconfig option)
namespace {
// Sums the CPU cycles of each operation type (TFLM calls BeginEvent/EndEvent around every operator)
class OpProfiler : public tflite::MicroProfilerInterface {
   public:
    uint32_t BeginEvent(const char *tag) override {
        if (open_ >= 4) return 0;
        tag_[open_] = tag;
        t0_[open_] = esp_cpu_get_cycle_count();
        return open_++;
    }
    void EndEvent(uint32_t h) override {
        const uint32_t dt = esp_cpu_get_cycle_count() - t0_[h];
        open_ = h;
        int k = 0;
        while (k < n_ && strcmp(names_[k], tag_[h]) != 0) k++;
        if (k == n_) {
            if (n_ == kMax) return;
            names_[n_++] = tag_[h];
        }
        cycles_[k] += dt;
        count_[k]++;
    }
    void print_and_reset(uint32_t inferences) {
        printf("[prof] model ops per inference (us @240MHz):");
        uint64_t tot = 0;
        for (int k = 0; k < n_; k++) {
            printf(" %s=%.1f/%lu", names_[k], cycles_[k] / 240.0 / inferences, (unsigned long)(count_[k] / inferences));
            tot += cycles_[k];
            cycles_[k] = count_[k] = 0;
        }
        printf(" | total=%.1f\n", tot / 240.0 / inferences);
    }

   private:
    static constexpr int kMax = 32;
    const char *tag_[4];
    uint32_t t0_[4];
    int open_ = 0, n_ = 0;
    const char *names_[kMax];
    uint64_t cycles_[kMax] = {};
    uint32_t count_[kMax] = {};
};
OpProfiler g_prof;
uint32_t g_prof_inferences = 0, g_prof_frames = 0;
}  // namespace
#define PROFILER_ARG , &g_prof
#else
#define PROFILER_ARG
#endif

namespace {

// ---- Feature settings. These MUST match training (microWakeWord / pymicro-features). ----
constexpr int kSampleRate = 16000;
constexpr int kFeatureSize = 40;         // 40 frequency bands
constexpr int kFeatureWindowMs = 30;     // each feature looks at 30 ms of audio
constexpr int kFeatureStepMs = 10;       // a new feature every 10 ms
constexpr float kFeatureScale = 0.0390625f;  // frontend integer output -> float units used in training
constexpr size_t kVarArenaSize = 1024;   // bookkeeping for the model's internal state variables
constexpr int kMaxWindow = 20;

FrontendConfig g_fcfg;
FrontendState g_fstate;

tflite::MicroMutableOpResolver<20> g_resolver;
const tflite::Model *g_model = nullptr;
uint8_t *g_arena = nullptr;
size_t g_arena_size = 0;
uint8_t *g_var_arena = nullptr;
tflite::MicroResourceVariables *g_mrv = nullptr;
tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_in = nullptr;
TfLiteTensor *g_out = nullptr;

int g_stride = 3;          // feature frames per inference
int g_step = 0;            // frames already placed in the input tensor
float g_in_mult = 0;       // frontend value -> int8 input:  q = round(v * g_in_mult) + g_in_zp
int g_in_zp = 0;
float g_out_scale = 1.0f / 256.0f;
int g_out_zp = 0;

float g_cutoff = 0.9f;
int g_window = 5;
float g_probs[kMaxWindow];
int g_prob_idx = 0;
int g_cooldown_frames = 100;
int g_ignore = 0;          // < 0: detection blocked (warm-up / cool-down)

portMUX_TYPE g_stats_lock = portMUX_INITIALIZER_UNLOCKED;
uint32_t g_stat_inferences = 0;
uint64_t g_stat_busy_us = 0;
uint64_t g_stat_invoke_us = 0;
float g_stat_max_avg = 0;
float g_last_avg = 0;

// Score events: one log line per rise of the averaged probability (real wake words AND near-misses), so the
// threshold can be chosen from real data. An event starts above 0.20 and ends after 5 inferences below 0.10.
constexpr float kEventStart = 0.20f, kEventEnd = 0.10f;
bool g_evt = false, g_evt_detected = false;
float g_evt_peak = 0;
int g_evt_runs = 0, g_evt_quiet = 0;
float g_last_evt_peak = 0;
int g_last_evt_ms = 0;

void track_event(float avg, bool detected) {
    if (!g_evt) {
        if (avg < kEventStart) return;
        g_evt = true;
        g_evt_detected = false;
        g_evt_peak = 0;
        g_evt_runs = g_evt_quiet = 0;
    }
    g_evt_peak = std::max(g_evt_peak, avg);
    g_evt_detected |= detected;
    g_evt_runs++;
    g_evt_quiet = avg < kEventEnd ? g_evt_quiet + 1 : 0;
    if (g_evt_quiet >= 5) {
        const int ms = (g_evt_runs - g_evt_quiet) * g_stride * kFeatureStepMs;
        portENTER_CRITICAL(&g_stats_lock);
        g_last_evt_peak = g_evt_peak;
        g_last_evt_ms = ms;
        portEXIT_CRITICAL(&g_stats_lock);
        ESP_LOGI(TAG, "score event: peak %.2f over %d ms -> %s", g_evt_peak,
                 ms,
                 g_evt_detected ? "detected" : "not detected (below threshold or in cool-down)");
        g_evt = false;
    }
}

bool register_ops() {
    // Superset of the operations used by microWakeWord streaming models (same list as ESPHome).
    return g_resolver.AddCallOnce() == kTfLiteOk && g_resolver.AddVarHandle() == kTfLiteOk &&
           g_resolver.AddReshape() == kTfLiteOk && g_resolver.AddReadVariable() == kTfLiteOk &&
           g_resolver.AddStridedSlice(KWS_STRIDED_SLICE) == kTfLiteOk &&
           g_resolver.AddConcatenation(KWS_CONCATENATION) == kTfLiteOk &&
           g_resolver.AddAssignVariable() == kTfLiteOk && g_resolver.AddConv2D() == kTfLiteOk &&
           g_resolver.AddMul() == kTfLiteOk && g_resolver.AddAdd() == kTfLiteOk &&
           g_resolver.AddMean() == kTfLiteOk && g_resolver.AddFullyConnected() == kTfLiteOk &&
           g_resolver.AddLogistic() == kTfLiteOk && g_resolver.AddQuantize() == kTfLiteOk &&
           g_resolver.AddDepthwiseConv2D() == kTfLiteOk && g_resolver.AddAveragePool2D() == kTfLiteOk &&
           g_resolver.AddMaxPool2D() == kTfLiteOk && g_resolver.AddPad() == kTfLiteOk &&
           g_resolver.AddPack() == kTfLiteOk && g_resolver.AddSplitV(KWS_SPLIT_V) == kTfLiteOk;
}

void reset_resource_variables() {
    // The state variables live in the tensor arena; their bookkeeping lives in the small variable arena.
    // After a failed allocation attempt the bookkeeping must be recreated.
    tflite::MicroAllocator *va = tflite::MicroAllocator::Create(g_var_arena, kVarArenaSize);
    g_mrv = tflite::MicroResourceVariables::Create(va, 20);
}

bool frontend_init() {
    g_fcfg.window.size_ms = kFeatureWindowMs;
    g_fcfg.window.step_size_ms = kFeatureStepMs;
    g_fcfg.filterbank.num_channels = kFeatureSize;
    g_fcfg.filterbank.lower_band_limit = 125.0f;
    g_fcfg.filterbank.upper_band_limit = 7500.0f;
    g_fcfg.noise_reduction.smoothing_bits = 10;
    g_fcfg.noise_reduction.even_smoothing = 0.025f;
    g_fcfg.noise_reduction.odd_smoothing = 0.06f;
    g_fcfg.noise_reduction.min_signal_remaining = 0.05f;
    g_fcfg.pcan_gain_control.enable_pcan = 1;
    g_fcfg.pcan_gain_control.strength = 0.95f;
    g_fcfg.pcan_gain_control.offset = 80.0f;
    g_fcfg.pcan_gain_control.gain_bits = 21;
    g_fcfg.log_scale.enable_log = 1;
    g_fcfg.log_scale.scale_shift = 6;
    return FrontendPopulateState(&g_fcfg, &g_fstate, kSampleRate) != 0;
}

void clear_probabilities() {
    for (float &p : g_probs) p = 0.0f;
    g_prob_idx = 0;
    g_ignore = -g_cooldown_frames;
}

// ---- Quiet-room gate: saves the model's CPU time while nothing is being said. ----
// The feature frontend runs on every 10 ms frame (it tracks the background noise), but the model only runs while
// the audio is above the background. After 1.6 s of quiet (longer than the model's ~1.5 s memory) inference pauses;
// when sound returns, the frames of the pause (up to 300 ms) are replayed through the model first, so the quiet
// start of a word ("M-") is not lost. In a quiet room this skips most inferences; in steady noise (fan, AC) too.
constexpr int kGateHoldFrames = 160;  // 1.6 s
constexpr int kLookbackFrames = 30;   // 300 ms
constexpr float kGateRatio = 2.0f;    // "sound" = a 20 ms block 6 dB above the background (3 dB: open 99% of the
                                      // time in a lab with a -44 dBFS floor; speech at 1 m is 15-25 dB above it)
float g_noise_rms = 3000.0f;          // background level (RMS): falls fast to the real level, rises ~10 % per second
int g_gate_hold = kGateHoldFrames;    // frames until the model pauses
bool g_gate_open = true;
int g_paused_frames = 0;              // frames skipped in the current pause (replayed when sound returns)
int8_t g_lookback[kLookbackFrames][kFeatureSize];
int g_lb_next = 0;
uint32_t g_stat_skipped_frames = 0;

void gate_reset() {
    g_noise_rms = 3000.0f;
    g_gate_hold = kGateHoldFrames;
    g_gate_open = true;
    g_paused_frames = 0;
    g_lb_next = 0;
}

// Runs the model on the frames in the input tensor and updates the averaged probability. true = detection.
bool infer(int64_t *invoke_us) {
    const int64_t t0 = esp_timer_get_time();
    const TfLiteStatus st = g_interp->Invoke();
    *invoke_us += esp_timer_get_time() - t0;
    if (st != kTfLiteOk) {
        ESP_LOGW(TAG, "Invoke failed");
        return false;
    }
    portENTER_CRITICAL(&g_stats_lock);
    g_stat_inferences++;
    portEXIT_CRITICAL(&g_stats_lock);
#if CONFIG_KWS_PROFILE_OPS
    if (++g_prof_inferences == 1000) {
        g_prof.print_and_reset(g_prof_inferences);
        static const char *stage[8] = {"window", "fft", "energy", "filterbank", "sqrt", "noise_red", "pcan", "log"};
        printf("[prof] feature stages per 10 ms frame (us):");
        for (int k = 0; k < 8; k++) {
            printf(" %s=%.1f", stage[k], g_frontend_prof_cycles[k] / 240.0 / g_prof_frames);
            g_frontend_prof_cycles[k] = 0;
        }
        printf("\n");
        g_prof_inferences = g_prof_frames = 0;
    }
#endif
    const float p = (g_out->data.uint8[0] - g_out_zp) * g_out_scale;
    g_probs[g_prob_idx] = p;
    g_prob_idx = (g_prob_idx + 1) % g_window;

    float sum = 0;
    for (int i = 0; i < g_window; i++) sum += g_probs[i];
    g_last_avg = sum / g_window;
    portENTER_CRITICAL(&g_stats_lock);
    if (g_last_avg > g_stat_max_avg) g_stat_max_avg = g_last_avg;
    portEXIT_CRITICAL(&g_stats_lock);

    const bool hit = g_ignore >= 0 && g_last_avg > g_cutoff;
    track_event(g_last_avg, hit);
    if (hit) clear_probabilities();  // start the cool-down, avoids double detections
    return hit;
}

// Puts one feature frame into the model input; runs the model when `stride` frames are in (every 30 ms)
bool feed_frame(const int8_t *frame, int64_t *invoke_us) {
    std::memcpy(g_in->data.int8 + g_step * kFeatureSize, frame, kFeatureSize);
    if (++g_step < g_stride) return false;
    g_step = 0;
    return infer(invoke_us);
}

void describe_input(const TfLiteTensor *t) {
    char dims[64] = {0};
    int off = 0;
    for (int i = 0; i < t->dims->size && off < (int)sizeof(dims) - 8; i++)
        off += snprintf(dims + off, sizeof(dims) - off, "%s%d", i ? "x" : "", t->dims->data[i]);
    ESP_LOGE(TAG, "  model input: [%s] type=%d", dims, (int)t->type);
}

}  // namespace

extern "C" esp_err_t ww_init(const uint8_t *model_data, size_t model_len, const ww_config_t *cfg, ww_info_t *info) {
    size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    if (model_data == nullptr || model_len < 16) return ESP_ERR_INVALID_ARG;
    flatbuffers::Verifier verifier(model_data, model_len);
    if (!tflite::VerifyModelBuffer(verifier)) {
        ESP_LOGE(TAG, "The model file is not a valid .tflite file (corrupted or wrong file?)");
        return ESP_ERR_INVALID_ARG;
    }
    const tflite::Model *model = tflite::GetModel(model_data);
    g_model = model;
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %lu is not supported (expected %d). Is this a .tflite file?",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }
    if (!register_ops()) {
        ESP_LOGE(TAG, "Could not register TFLite operations");
        return ESP_FAIL;
    }
    if (!frontend_init()) {
        ESP_LOGE(TAG, "Could not start the audio feature generator (out of memory?)");
        return ESP_ERR_NO_MEM;
    }

    g_var_arena = (uint8_t *)heap_caps_aligned_alloc(16, kVarArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!g_var_arena) return ESP_ERR_NO_MEM;

    // Try the configured arena size, then grow it. Internal RAM is used because it is fastest.
    const size_t base = std::max<size_t>(cfg->tensor_arena_bytes, 8 * 1024);
    const size_t attempts[] = {base, base * 3 / 2, base * 2, base * 3};
    for (size_t attempt : attempts) {
        attempt = (attempt + 15) & ~static_cast<size_t>(15);
        uint8_t *arena = (uint8_t *)heap_caps_aligned_alloc(16, attempt, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!arena) continue;
        reset_resource_variables();
        auto *interp = new tflite::MicroInterpreter(model, g_resolver, arena, attempt, g_mrv PROFILER_ARG);
        if (interp->AllocateTensors() == kTfLiteOk) {
            g_arena = arena;
            g_arena_size = attempt;
            g_interp = interp;
            break;
        }
        delete interp;
        heap_caps_free(arena);
        ESP_LOGW(TAG, "Tensor arena of %u bytes is too small, trying bigger", (unsigned)attempt);
    }
    if (!g_interp) {
        ESP_LOGE(TAG, "Could not allocate the model (unsupported operation or not enough RAM). See messages above.");
        return ESP_ERR_NO_MEM;
    }

    g_in = g_interp->input(0);
    g_out = g_interp->output(0);

    // A microWakeWord streaming model takes [1, stride, 40] int8 features and returns [1, 1] uint8.
    const bool in_ok = g_in->dims->size == 3 && g_in->dims->data[0] == 1 && g_in->dims->data[2] == kFeatureSize &&
                       g_in->type == kTfLiteInt8 && g_in->dims->data[1] >= 1 && g_in->dims->data[1] <= 10;
    const bool out_ok = g_out->dims->size == 2 && g_out->dims->data[0] == 1 && g_out->dims->data[1] == 1 &&
                        g_out->type == kTfLiteUInt8;
    if (!in_ok || !out_ok) {
        ESP_LOGE(TAG, "This is NOT a streaming microWakeWord model, so this firmware cannot run it.");
        describe_input(g_in);
        ESP_LOGE(TAG, "  expected input [1x3x40] int8 and output [1x1] uint8.");
        ESP_LOGE(TAG, "  (A 1-second classifier with input like [1x98x40x1] needs a different feature pipeline.)");
        return ESP_ERR_NOT_SUPPORTED;
    }

    g_stride = g_in->dims->data[1];
    g_in_mult = kFeatureScale / g_in->params.scale;
    g_in_zp = g_in->params.zero_point;
    g_out_scale = g_out->params.scale > 0 ? g_out->params.scale : 1.0f / 256.0f;
    g_out_zp = g_out->params.zero_point;

    g_cutoff = cfg->probability_cutoff;
    g_window = std::min(std::max(cfg->sliding_window, 1), kMaxWindow);
    g_cooldown_frames = std::max(cfg->cooldown_ms / kFeatureStepMs, 0);
    std::memset(g_in->data.int8, g_in_zp, g_in->bytes);
    g_step = 0;
    clear_probabilities();

    if (info) {
        info->input_frames = g_stride;
        info->feature_size = kFeatureSize;
        info->input_scale = g_in->params.scale;
        info->input_zero_point = g_in_zp;
        info->arena_size = g_arena_size;
        info->arena_used = g_interp->arena_used_bytes();
        info->heap_used = heap_before - heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    }
    return ESP_OK;
}

extern "C" bool ww_process(const int16_t *samples, size_t num_samples, float *avg_probability) {
    if (!g_interp) return false;
    const int64_t t0 = esp_timer_get_time();
    int64_t invoke_us = 0;
    bool detected = false;

    // Quiet-room gate: sound above the background keeps the model running for another 1.6 s
    int64_t ss = 0;
    for (size_t i = 0; i < num_samples; i++) ss += (int32_t)samples[i] * samples[i];
    const float rms = num_samples ? sqrtf((float)ss / (float)num_samples) : 0.0f;
    g_noise_rms = rms < g_noise_rms ? 0.9f * g_noise_rms + 0.1f * rms : g_noise_rms * 1.002f + 0.01f;
    if (g_noise_rms < 1.0f) g_noise_rms = 1.0f;
    if (rms > g_noise_rms * kGateRatio) g_gate_hold = kGateHoldFrames;

    size_t pos = 0;
    while (pos < num_samples) {
        size_t read = 0;
        FrontendOutput f = FrontendProcessSamples(&g_fstate, samples + pos, num_samples - pos, &read);
        pos += read;
        if (f.size == 0) {
            if (read == 0) break;
            continue;
        }

#if CONFIG_KWS_PROFILE_OPS
        g_prof_frames++;
#endif
        // A new 40-value feature frame (every 10 ms): quantize it exactly like the model's input expects
        int8_t *frame = g_lookback[g_lb_next];
        g_lb_next = (g_lb_next + 1) % kLookbackFrames;
        for (size_t i = 0; i < f.size && i < (size_t)kFeatureSize; i++) {
            int32_t q = (int32_t)lrintf((float)f.values[i] * g_in_mult) + g_in_zp;
            frame[i] = (int8_t)std::min<int32_t>(127, std::max<int32_t>(-128, q));
        }

        if (g_gate_hold > 0) {
            g_gate_hold--;
            // Sound again after a pause: first the paused frames (up to 300 ms, oldest first), then this one
            const int replay = g_gate_open ? 0 : std::min(g_paused_frames, kLookbackFrames - 1);
            for (int k = replay; k >= 0; k--)
                detected |= feed_frame(g_lookback[(g_lb_next - 1 - k + 2 * kLookbackFrames) % kLookbackFrames], &invoke_us);
            g_gate_open = true;
            g_paused_frames = 0;
        } else {
            if (g_gate_open) g_step = 0;  // a replay starts on a clean model input
            g_gate_open = false;
            g_paused_frames++;
            portENTER_CRITICAL(&g_stats_lock);
            g_stat_skipped_frames++;
            portEXIT_CRITICAL(&g_stats_lock);
        }

        // Cool-down counts only frames where the model is below the threshold (same idea as ESPHome)
        const float latest = g_probs[(g_prob_idx + g_window - 1) % g_window];
        if (g_ignore < 0 && latest < g_cutoff) g_ignore++;
    }

    const uint64_t busy = (uint64_t)(esp_timer_get_time() - t0);
    portENTER_CRITICAL(&g_stats_lock);
    g_stat_busy_us += busy;
    g_stat_invoke_us += (uint64_t)invoke_us;
    portEXIT_CRITICAL(&g_stats_lock);
    if (avg_probability) *avg_probability = g_last_avg;
    return detected;
}

extern "C" void ww_reset(void) {
    if (!g_interp) return;
    // Rebuild the interpreter on the same memory: all internal state starts exactly as after boot.
    delete g_interp;
    reset_resource_variables();
    g_interp = new tflite::MicroInterpreter(g_model, g_resolver, g_arena, g_arena_size, g_mrv PROFILER_ARG);
    if (g_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "model reset failed");
        delete g_interp;
        g_interp = nullptr;
        return;
    }
    g_in = g_interp->input(0);
    g_out = g_interp->output(0);
    FrontendReset(&g_fstate);
    std::memset(g_in->data.int8, g_in_zp, g_in->bytes);
    g_step = 0;
    g_last_avg = 0;
    g_evt = false;
    gate_reset();
    clear_probabilities();
}

extern "C" void ww_last_event(float *peak, int *ms) {
    portENTER_CRITICAL(&g_stats_lock);
    *peak = g_last_evt_peak;
    *ms = g_last_evt_ms;
    portEXIT_CRITICAL(&g_stats_lock);
}

extern "C" void ww_take_stats(ww_stats_t *out) {
    portENTER_CRITICAL(&g_stats_lock);
    out->inferences = g_stat_inferences;
    out->busy_us = g_stat_busy_us;
    out->invoke_us = g_stat_invoke_us;
    out->skipped = g_stat_skipped_frames / (uint32_t)g_stride;
    out->max_avg_probability = g_stat_max_avg;
    out->last_avg_probability = g_last_avg;
    g_stat_inferences = 0;
    g_stat_busy_us = 0;
    g_stat_invoke_us = 0;
    g_stat_skipped_frames = 0;
    g_stat_max_avg = 0;
    portEXIT_CRITICAL(&g_stats_lock);
}
