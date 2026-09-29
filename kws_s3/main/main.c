// SIH KWS firmware (ESP32-S3 + INMP441 microphone)
//
//  mic --I2S/DMA--> audio task (core 1): filter -> 2 s mu-law ring ---------------------+
//                                        \-> wake_word (features + streaming model)     |
//                                                  | detection (LED blink)              v
//                                                  +--> streamer (core 0) --WebSocket--> ASR server
//                                                       (only what is said after the wake word)
//
// Everything is configured with "idf.py menuconfig" -> "KWS (wake word) settings".
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#include "audio_input.h"
#include "status_led.h"
#include "streamer.h"
#include "wake_word.h"
#include "wav_selftest.h"
#include "wifi.h"

static const char *TAG = "kws";

#ifndef CONFIG_KWS_INJECT_TEST
#define CONFIG_KWS_INJECT_TEST 0
#endif

// Generated from the "model" folder by main/embed_model.cmake
extern const char g_ww_model_file[];
extern const unsigned char g_ww_model[];
extern const size_t g_ww_model_len;
extern const unsigned char g_ww_manifest[];
extern const size_t g_ww_manifest_len;
extern const unsigned char g_ww_selftest[];
extern const size_t g_ww_selftest_len;

// Linker symbols: static data (.data + .bss) in internal RAM, and code copied into IRAM
extern uint8_t _data_start, _bss_end, _iram_start, _iram_end;

typedef struct {
    char wake_word[48];
    float cutoff;
    int window;
    int arena_bytes;
    bool from_json;
} model_settings_t;

static bool s_model_ok;
static bool s_streaming_enabled;
static model_settings_t s_ms;
static volatile uint32_t s_detections;
// Detections are logged and blinked by the main task: the audio task never waits on the console or the LED
static TaskHandle_t s_main_task;
static volatile float s_det_score;
static volatile int64_t s_det_us;
static volatile bool s_det_extended;
static size_t s_ballast;  // RAM reserved by the KWS_RAM_LIMIT_KB test (not used by the firmware)
static volatile uint32_t s_alloc_failures;  // heap allocations that failed since boot (e.g. Wi-Fi buffers in a burst)
static volatile uint32_t s_alloc_fail_bytes;
static bool s_mics_both, s_mics_aligned;  // two microphones (time-aligned): the status line shows their mix and delay

// ---------------------------------------------------------------------------------------------------------------
static size_t static_ram(void) { return (size_t)(&_bss_end - &_data_start); }
static size_t iram_code(void) { return (size_t)(&_iram_end - &_iram_start); }  // same SRAM, counted for the SIH limit

// RAM the firmware uses: static data + heap in use (now, and the highest since boot). The ballast is not counted.
static void ram_used(size_t *now, size_t *peak) {
    const size_t total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    *now = static_ram() + total - heap_caps_get_free_size(MALLOC_CAP_INTERNAL) - s_ballast;
    *peak = static_ram() + total - heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) - s_ballast;
}

#if CONFIG_KWS_RAM_LIMIT_KB > 0
// SIH limit test: reserve every byte of internal RAM beyond CONFIG_KWS_RAM_LIMIT_KB (counted strictly: IRAM code +
// static data + heap), so the rest of the firmware (Wi-Fi, model, audio, streaming) has to live within the limit.
// Never freed.
static void apply_ram_limit(void) {
    size_t now, peak;
    ram_used(&now, &peak);
    const size_t limit = (size_t)CONFIG_KWS_RAM_LIMIT_KB * 1024 - iram_code();  // what is left for data
    if (now >= limit) {
        ESP_LOGE(TAG, "RAM limit test: already %u KB used at boot, above the %u KB limit", (unsigned)(now / 1024),
                 (unsigned)(limit / 1024));
        return;
    }
    const size_t free_now = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t want = free_now > limit - now ? free_now - (limit - now) : 0;
    while (want > 64) {  // several blocks: the free RAM is split over a few regions
        size_t block = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
        if (block > want) block = want;
        if (block < 64 || !heap_caps_malloc(block, MALLOC_CAP_INTERNAL)) break;
        s_ballast += block;
        want -= block;
    }
    ESP_LOGW(TAG, "RAM limit test: %u KB reserved, the firmware has %u KB (static data + heap) + %u KB IRAM code "
                  "= %d KB to work with", (unsigned)(s_ballast / 1024), (unsigned)(limit / 1024),
             (unsigned)(iram_code() / 1024), CONFIG_KWS_RAM_LIMIT_KB);
}
#endif

static void on_alloc_failed(size_t size, uint32_t caps, const char *function_name) {
    s_alloc_failures++;  // counted only: this can run in any task, even inside the Wi-Fi driver
    s_alloc_fail_bytes = size;
}

static const char *reset_reason(void) {
    switch (esp_reset_reason()) {
        case ESP_RST_POWERON: return "power-on";
        case ESP_RST_EXT: return "reset pin";
        case ESP_RST_SW: return "software restart";
        case ESP_RST_PANIC: return "crash (panic)";
        case ESP_RST_INT_WDT: return "interrupt watchdog";
        case ESP_RST_TASK_WDT: return "task watchdog (a task stopped running)";
        case ESP_RST_WDT: return "watchdog";
        case ESP_RST_BROWNOUT: return "brownout (supply voltage dropped)";
        case ESP_RST_USB: return "USB";
        default: return "other";
    }
}

static void print_banner(void) {
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    uint32_t flash = 0;
    esp_flash_get_size(NULL, &flash);
    printf("\n==================== SIH KWS firmware ====================\n");
    printf(" chip      : %s rev v%d.%d, %d cores, flash %lu MB\n", CONFIG_IDF_TARGET, chip.revision / 100,
           chip.revision % 100, chip.cores, (unsigned long)(flash >> 20));
    printf(" ESP-IDF   : %s\n", esp_get_idf_version());
    printf(" reset     : %s\n", reset_reason());
    printf(" RAM       : %u KB static data, %u KB heap free (code in IRAM: %u KB, counted in the SIH total)\n",
           (unsigned)(static_ram() / 1024), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
           (unsigned)(iram_code() / 1024));
    printf(" model     : %s (%u bytes, stored in flash)\n", g_ww_model_len ? g_ww_model_file : "none",
           (unsigned)g_ww_model_len);
    printf("===========================================================\n\n");
#if !CONFIG_IDF_TARGET_ESP32S3
    ESP_LOGW(TAG, "This firmware is tuned for the ESP32-S3 but was built for %s", CONFIG_IDF_TARGET);
#endif
}

// Reads the model's .json manifest (threshold, window, name, arena) and applies menuconfig overrides.
static void load_model_settings(model_settings_t *m) {
    memset(m, 0, sizeof(*m));
    m->cutoff = 0.90f;
    m->window = 5;
    m->arena_bytes = 30000;  // microWakeWord models; the arena grows automatically if the model needs more
    // default name = model file name without ".tflite"
    strlcpy(m->wake_word, g_ww_model_len ? g_ww_model_file : "none", sizeof(m->wake_word));
    char *dot = strrchr(m->wake_word, '.');
    if (dot) *dot = 0;

    if (g_ww_manifest_len > 0) {
        cJSON *root = cJSON_ParseWithLength((const char *)g_ww_manifest, g_ww_manifest_len);
        if (!root) {
            ESP_LOGW(TAG, "model .json could not be parsed, using defaults");
        } else {
            const cJSON *ww = cJSON_GetObjectItem(root, "wake_word");
            if (cJSON_IsString(ww)) strlcpy(m->wake_word, ww->valuestring, sizeof(m->wake_word));
            const cJSON *micro = cJSON_GetObjectItem(root, "micro");
            const cJSON *v;
            if ((v = cJSON_GetObjectItem(micro, "probability_cutoff")) && cJSON_IsNumber(v)) m->cutoff = (float)v->valuedouble;
            if ((v = cJSON_GetObjectItem(micro, "sliding_window_size")) && cJSON_IsNumber(v)) m->window = v->valueint;
            if ((v = cJSON_GetObjectItem(micro, "tensor_arena_size")) && cJSON_IsNumber(v) && v->valueint > 0)
                m->arena_bytes = v->valueint;
            if ((v = cJSON_GetObjectItem(micro, "feature_step_size")) && cJSON_IsNumber(v) && v->valueint != 10)
                ESP_LOGE(TAG, "model uses %d ms feature steps; this firmware needs a microWakeWord v2 model (10 ms)",
                         v->valueint);
            m->from_json = true;
            cJSON_Delete(root);
        }
    } else if (g_ww_model_len) {
        ESP_LOGW(TAG, "no .json manifest in the model folder, using threshold %.2f and window %d", m->cutoff, m->window);
    }
    if (CONFIG_KWS_CUTOFF_PERCENT > 0) m->cutoff = CONFIG_KWS_CUTOFF_PERCENT / 100.0f;
    if (CONFIG_KWS_SLIDING_WINDOW > 0) m->window = CONFIG_KWS_SLIDING_WINDOW;
    if (CONFIG_KWS_TENSOR_ARENA_KB > 0) m->arena_bytes = CONFIG_KWS_TENSOR_ARENA_KB * 1024;
}

static void run_selftest(void) {
    selftest_result_t pos, neg;
    wav_selftest_run(g_ww_selftest_len ? g_ww_selftest : NULL, g_ww_selftest_len, &pos, &neg);
    printf("\n------------------------ SELF-TEST ------------------------\n");
    printf(" noise only (3 s)   : %s  (highest score %.2f, threshold %.2f)\n",
           neg.detected ? "FAIL - false detection!" : "PASS - stayed quiet", neg.max_score, s_ms.cutoff);
    if (pos.ran) {
        printf(" wake word recording: %s  (highest score %.2f", pos.detected ? "PASS - detected" : "FAIL - not detected",
               pos.max_score);
        if (pos.detected) printf(", fired %d ms after the recording started; recording is %d ms", pos.detect_ms, pos.clip_ms);
        printf(")\n");
    } else {
        printf(" wake word recording: skipped - %s\n", pos.error);
    }
    printf("-----------------------------------------------------------\n\n");
}

// ---------------------------------------------------------------------------------------------------------------
// Capture + detection in one task on core 1 (away from Wi-Fi on core 0). It blocks on the I2S DMA buffer.
// A detection starts the stream first; the main task then logs it and blinks the LED.
static void audio_task(void *arg) {
    static int16_t block[AUDIO_BLOCK_SAMPLES];
    int no_data_s = 0;
    esp_task_wdt_add(NULL);  // a stuck audio task restarts the board (task watchdog, 5 s)
    for (;;) {
        const size_t n = audio_input_read(block);  // waits at most 1 s
        esp_task_wdt_reset();
        if (!n) {  // the I2S DMA stopped (a missing microphone still delivers zeros): start over
            if (++no_data_s >= 5) {
                ESP_LOGE(TAG, "no audio from I2S for 5 s: restarting");
                esp_restart();
            }
            continue;
        }
        no_data_s = 0;
        float score = 0;
        if (!s_model_ok || !ww_process(block, n, &score)) continue;

        s_det_extended = s_streaming_enabled && streamer_is_active();
        if (s_streaming_enabled) streamer_trigger(score, "wake_word");
        s_det_score = score;
        s_det_us = esp_timer_get_time();
        s_detections++;
        xTaskNotifyGive(s_main_task);
    }
}

// CPU load of each core from the FreeRTOS idle tasks (percent, since the previous call)
static void cpu_load(float load[2]) {
    load[0] = load[1] = -1;
#if configUSE_TRACE_FACILITY && configGENERATE_RUN_TIME_STATS
    static configRUN_TIME_COUNTER_TYPE prev_total, prev_idle[2];
    UBaseType_t cap = uxTaskGetNumberOfTasks() + 4;
    TaskStatus_t *st = malloc(cap * sizeof(TaskStatus_t));
    if (!st) return;
    configRUN_TIME_COUNTER_TYPE total = 0, idle[2] = {0, 0};
    UBaseType_t n = uxTaskGetSystemState(st, cap, &total);
    for (UBaseType_t i = 0; i < n; i++)
        for (int c = 0; c < portNUM_PROCESSORS && c < 2; c++)
            if (st[i].xHandle == xTaskGetIdleTaskHandleForCore(c)) idle[c] = st[i].ulRunTimeCounter;
    free(st);
    configRUN_TIME_COUNTER_TYPE dt = total - prev_total;
    if (prev_total && dt) {
        for (int c = 0; c < 2; c++) {
            float idle_pct = 100.0f * (float)(idle[c] - prev_idle[c]) / (float)dt;
            load[c] = idle_pct > 100 ? 0 : 100.0f - idle_pct;
        }
    }
    prev_total = total;
    prev_idle[0] = idle[0];
    prev_idle[1] = idle[1];
#endif
}

// Every task's CPU share since the previous call (% of one core) and free stack (bytes never used since boot)
static void print_tasks(void) {
#if configUSE_TRACE_FACILITY && configGENERATE_RUN_TIME_STATS
    enum { MAXT = 32 };
    static TaskHandle_t prev_h[MAXT];
    static configRUN_TIME_COUNTER_TYPE prev_rt[MAXT], prev_total;
    static int prev_n;
    UBaseType_t cap = uxTaskGetNumberOfTasks() + 4;
    TaskStatus_t *st = malloc(cap * sizeof(TaskStatus_t));
    if (!st) return;
    configRUN_TIME_COUNTER_TYPE total = 0;
    UBaseType_t n = uxTaskGetSystemState(st, cap, &total);
    const float dt = prev_total ? (float)(total - prev_total) : 0.0f;
    printf("[tasks] %% of one core (free stack):");
    for (UBaseType_t i = 0; i < n; i++) {
        float pct = -1;
        for (int k = 0; k < prev_n; k++)
            if (prev_h[k] == st[i].xHandle && dt > 0) pct = 100.0f * (float)(st[i].ulRunTimeCounter - prev_rt[k]) / dt;
        const BaseType_t core = xTaskGetCoreID(st[i].xHandle);
        printf(" %s@%c=%.2f(%u)", st[i].pcTaskName, core == tskNO_AFFINITY ? '*' : (char)('0' + core), pct,
               (unsigned)st[i].usStackHighWaterMark);
    }
    printf("\n");
    prev_n = 0;
    for (UBaseType_t i = 0; i < n && i < MAXT; i++, prev_n++) {
        prev_h[i] = st[i].xHandle;
        prev_rt[i] = st[i].ulRunTimeCounter;
    }
    prev_total = total;
    free(st);
#endif
}

// Boot-time heap accounting: internal heap taken by each start-up stage
static void heap_mark(const char *stage) {
    static size_t last;
    const size_t free_now = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    if (last) printf("[heap] %-28s took %6d B, %6u B free\n", stage, (int)(last - free_now), (unsigned)free_now);
    else printf("[heap] %-28s %6u B free (of %u B)\n", stage, (unsigned)free_now,
                (unsigned)heap_caps_get_total_size(MALLOC_CAP_INTERNAL));
    last = free_now;
}

// The engine/microphone counters are taken (and reset) once per tick. With KWS_TELEMETRY_MS > 0 every tick also
// prints one "@T key=value ..." line for tools/dashboard.py (plus an "@I" info line every second); the human status
// line sums the ticks of its interval, so it reads the same with telemetry on or off.
typedef struct {
    int64_t us;                       // time covered
    int ticks, load_n;
    double mic_pow;                   // sum of the ticks' mean-square levels (linear), for the average dBFS
    float mic_peak_dbfs, score_max, load_sum[2], mic_delay_us, mic_left_share, mic_noise_db;
    uint32_t inferences, skipped, mic_delay_updates;
    uint64_t ww_busy_us, ww_invoke_us, audio_busy_us;
} status_acc_t;

static status_acc_t s_acc = {.mic_peak_dbfs = -120.0f};
static uint32_t s_lost_i2s, s_lost_net;  // audio lost since boot: I2S DMA (task too slow), network (too slow)

static void stats_tick(int64_t dt_us) {
    audio_stats_t a;
    audio_input_take_stats(&a);
    ww_stats_t w = {0};
    if (s_model_ok) ww_take_stats(&w);
    float load[2];
    cpu_load(load);
    s_lost_i2s += a.dma_overflows;
    s_lost_net += a.net_drops;

    s_acc.us += dt_us;
    s_acc.ticks++;
    s_acc.mic_pow += pow(10.0, a.level_dbfs / 10.0);
    s_acc.mic_peak_dbfs = fmaxf(s_acc.mic_peak_dbfs, a.peak_dbfs);
    s_acc.score_max = fmaxf(s_acc.score_max, w.max_avg_probability);
    s_acc.inferences += w.inferences;
    s_acc.skipped += w.skipped;
    s_acc.ww_busy_us += w.busy_us;
    s_acc.ww_invoke_us += w.invoke_us;
    s_acc.audio_busy_us += a.busy_us;
    s_acc.mic_delay_us = a.mic_delay_us;
    s_acc.mic_delay_updates = a.mic_delay_updates;
    s_acc.mic_left_share = a.mic_left_share;
    s_acc.mic_noise_db = a.mic_noise_db;
    if (load[0] >= 0) {  // -1 on the very first call (no previous counters yet)
        s_acc.load_sum[0] += load[0];
        s_acc.load_sum[1] += load[1];
        s_acc.load_n++;
    }

#if CONFIG_KWS_TELEMETRY_MS > 0
    size_t ram_now, ram_peak;
    ram_used(&ram_now, &ram_peak);
    // One printf per line: other tasks' log lines cannot land in the middle of it
    printf("@T t=%lld s=%.3f lv=%.1f pk=%.1f c0=%.1f c1=%.1f kw=%.2f inf=%.2f iv=%.2f n=%lu d=%lu "
           "sk=%lu hf=%u hm=%u ht=%u pf=0 pm=0 pt=0 ru=%u rp=%u\n",
           (long long)(esp_timer_get_time() / 1000), w.max_avg_probability, a.level_dbfs, a.peak_dbfs, load[0],
           load[1], 100.0f * (float)(w.busy_us + a.busy_us) / (float)dt_us,
           w.inferences + w.skipped ? w.busy_us / 1000.0f / (w.inferences + w.skipped) : 0.0f,
           w.inferences ? w.invoke_us / 1000.0f / w.inferences : 0.0f, (unsigned long)w.inferences,
           (unsigned long)s_detections, (unsigned long)w.skipped, (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) + s_ballast),
           (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) + s_ballast),
           (unsigned)heap_caps_get_total_size(MALLOC_CAP_INTERNAL), (unsigned)ram_now, (unsigned)ram_peak);
    static uint32_t n_ticks;
    const uint32_t info_every = CONFIG_KWS_TELEMETRY_MS >= 1000 ? 1 : 1000 / CONFIG_KWS_TELEMETRY_MS;
    if (n_ticks++ % info_every == 0)  // wifi/server: -1 = streaming off
        printf("@I ww=%s cut=%.2f win=%d cool=%d tick=%d wifi=%d srv=%d li=%lu ln=%lu\n", s_ms.wake_word,
               s_ms.cutoff, s_ms.window, CONFIG_KWS_COOLDOWN_MS, CONFIG_KWS_TELEMETRY_MS,
               s_streaming_enabled ? wifi_is_connected() : -1, s_streaming_enabled ? streamer_is_connected() : -1,
               (unsigned long)s_lost_i2s, (unsigned long)s_lost_net);
#endif
}

static void print_status(void) {
    const status_acc_t *s = &s_acc;
    const float us = s->us ? (float)s->us : 1.0f;
    const float kws_pct = 100.0f * (float)(s->ww_busy_us + s->audio_busy_us) / us;
    const float level_dbfs = s->ticks ? (float)(10.0 * log10(s->mic_pow / s->ticks)) : -120.0f;
    const float load0 = s->load_n ? s->load_sum[0] / s->load_n : -1, load1 = s->load_n ? s->load_sum[1] / s->load_n : -1;
    // per 30 ms of audio: the features always run, the model only while it is not paused
    const uint32_t slots = s->inferences + s->skipped;
    const float model_ms = s->inferences ? s->ww_invoke_us / 1000.0f / s->inferences : 0.0f;
    const float feat_ms = slots ? (s->ww_busy_us - s->ww_invoke_us) / 1000.0f / slots : 0.0f;
    size_t ram_now, ram_peak;
    ram_used(&ram_now, &ram_peak);

    printf("[status] up %llus | mic %5.1f dBFS (peak %5.1f) | score max %.2f | detections %lu | inferences %lu "
           "(%.1f/s; model %.2f ms each, paused in quiet %.0f%%; features %.2f ms per 30 ms) | CPU core0 %4.1f%% core1 %4.1f%% "
           "(wake word pipeline %4.2f%% of one core) | RAM used %u KB (peak %u KB; peak + IRAM code %u KB",
           (unsigned long long)(esp_timer_get_time() / 1000000), level_dbfs, s->mic_peak_dbfs, s->score_max,
           (unsigned long)s_detections, (unsigned long)s->inferences, s->inferences * 1e6f / us, model_ms,
           slots ? 100.0f * s->skipped / slots : 0.0f, feat_ms, load0, load1, kws_pct, (unsigned)(ram_now / 1024),
           (unsigned)(ram_peak / 1024), (unsigned)((ram_peak + iram_code()) / 1024));
    if (CONFIG_KWS_RAM_LIMIT_KB > 0) printf(" of the %d KB limit", CONFIG_KWS_RAM_LIMIT_KB);
    printf(", %u KB free)", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    if (s_streaming_enabled)
        printf(" | wifi %s, server %s", wifi_is_connected() ? "OK" : "--", streamer_is_connected() ? "OK" : "--");
    if (s_mics_both)
        printf(" | mics: mix left %.0f%% right %.0f%% (right noise %+.1f dB)", 100.0f * s->mic_left_share,
               100.0f * (1.0f - s->mic_left_share), s->mic_noise_db);
    if (s_mics_aligned) {  // path difference (0.343 mm per us) / spacing = sine of the talker's angle off the front
        const float sine = fmaxf(-1.0f, fminf(1.0f, s->mic_delay_us * 0.343f / CONFIG_KWS_MIC_SPACING_MM));
        printf(", right %+.0f us after left (talker %.0f deg to the %s), %lu estimates", s->mic_delay_us,
               fabsf(asinf(sine)) * 57.2958f, sine >= 0 ? "left" : "right", (unsigned long)s->mic_delay_updates);
    }
    printf(" | audio lost since boot: i2s %lu net %lu", (unsigned long)s_lost_i2s, (unsigned long)s_lost_net);
    if (s_alloc_failures)
        printf(" | failed allocations %lu (last %lu B)", (unsigned long)s_alloc_failures, (unsigned long)s_alloc_fail_bytes);
    printf("\n");
    if (s->mic_peak_dbfs < -85.0f) ESP_LOGW(TAG, "microphone is silent - check the wiring");
    static bool noise_warned;  // once: a healthy pair is within a few dB
    if (s_mics_both && !noise_warned && fabsf(s->mic_noise_db) > 10.0f) {
        const bool right = s->mic_noise_db > 0;
        ESP_LOGW(TAG, "the %s microphone is %.0f dB noisier than the %s one (electrical, not sound), so it gets only "
                      "%.0f%% of the mix. Check its VDD/GND/SD wires and that its L/R pin is firmly on %s",
                 right ? "right" : "left", fabsf(s->mic_noise_db), right ? "left" : "right",
                 100.0f * (right ? 1.0f - s->mic_left_share : s->mic_left_share), right ? "3V3" : "GND");
        noise_warned = true;
    }
    s_acc = (status_acc_t){.mic_peak_dbfs = -120.0f};
}

#if CONFIG_KWS_INJECT_TEST
// [DEVICE-INJECTED] benchmark: test audio arrives on the console UART and goes through ww_process() exactly like
// microphone audio (same 20 ms blocks). One clip = "INJ <id> <samples>\n" + samples x int16 LE; the detector is reset
// before every clip. Answer: "INJ <id> det=<detections> first=<sample index of the 1st> max=<highest average score>".
static void inject_loop(void) {
    const uart_port_t u = CONFIG_ESP_CONSOLE_UART_NUM;
    printf("INJECT READY 921600\n");
    fflush(stdout);
    uart_wait_tx_done(u, pdMS_TO_TICKS(200));
    uart_set_baudrate(u, 921600);
    static int16_t block[AUDIO_BLOCK_SAMPLES];
    char line[64];
    for (;;) {
        int n = 0;
        while (n < (int)sizeof(line) - 1) {  // header line
            uint8_t c;
            if (uart_read_bytes(u, &c, 1, portMAX_DELAY) != 1) continue;
            if (c == '\n') break;
            line[n++] = (char)c;
        }
        line[n] = 0;
        unsigned id = 0, total = 0;
        if (sscanf(line, "INJ %u %u", &id, &total) != 2) continue;
        ww_reset();
        ww_stats_t st;
        ww_take_stats(&st);
        unsigned fed = 0, dets = 0;
        int first = -1;
        while (fed < total) {
            const unsigned k = total - fed < AUDIO_BLOCK_SAMPLES ? total - fed : AUDIO_BLOCK_SAMPLES;
            int got = 0;
            while (got < (int)(k * 2)) got += uart_read_bytes(u, (uint8_t *)block + got, k * 2 - got, portMAX_DELAY);
            fed += k;
            if (ww_process(block, k, NULL)) {
                if (first < 0) first = (int)fed;
                dets++;
            }
        }
        ww_take_stats(&st);
        printf("INJ %u det=%u first=%d max=%.3f\n", id, dets, first, st.max_avg_probability);
        fflush(stdout);
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------------
void app_main(void) {
    // Console through the UART driver: printf copies into a buffer and returns. Without the driver every printf
    // busy-waits on the 128-byte UART FIFO (the 10 Hz telemetry line alone cost ~3% of core 0).
    if (uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, CONFIG_KWS_INJECT_TEST ? 16384 : 256, 1024, 0, NULL, 0) == ESP_OK)
        uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
#if CONFIG_KWS_RAM_LIMIT_KB > 0
    apply_ram_limit();
#endif
    heap_caps_register_failed_alloc_callback(on_alloc_failed);
    print_banner();
    heap_mark("start of app_main");
    status_led_init(CONFIG_KWS_LED_GPIO);

    esp_err_t err = ESP_OK;
#if CONFIG_ESP_WIFI_NVS_ENABLED || CONFIG_ESP_PHY_CALIBRATION_AND_DATA_STORAGE
    // Only Wi-Fi settings / PHY calibration data would live in NVS. The default build keeps neither (RAM limit):
    // the radio calibrates at every boot and the settings come from menuconfig.
    err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    heap_mark("NVS");
#endif

    // 1) Model -----------------------------------------------------------------------------------------------
    load_model_settings(&s_ms);
    if (g_ww_model_len) {
        ww_config_t wcfg = {
            .probability_cutoff = s_ms.cutoff,
            .sliding_window = s_ms.window,
            .cooldown_ms = CONFIG_KWS_COOLDOWN_MS,
            .tensor_arena_bytes = (size_t)s_ms.arena_bytes,
        };
        ww_info_t info;
        err = ww_init(g_ww_model, g_ww_model_len, &wcfg, &info);
        if (err == ESP_OK) {
            s_model_ok = true;
            ESP_LOGI(TAG, "model ready: wake word \"%s\", threshold %.2f (%s), window %d, input %dx%d int8",
                     s_ms.wake_word, s_ms.cutoff, CONFIG_KWS_CUTOFF_PERCENT ? "menuconfig" : (s_ms.from_json ? "json" : "default"),
                     s_ms.window, info.input_frames, info.feature_size);
            ESP_LOGI(TAG, "model RAM: tensor arena %u of %u bytes used; engine total %u KB of heap",
                     (unsigned)info.arena_used, (unsigned)info.arena_size, (unsigned)(info.heap_used / 1024));
#if CONFIG_KWS_SELFTEST
            run_selftest();
#endif
        } else {
            ESP_LOGE(TAG, "model could not be started (%s) -> running in microphone-test mode", esp_err_to_name(err));
            status_led_set(40, 0, 0);
        }
    } else {
        ESP_LOGW(TAG, "no model in the firmware -> microphone-test mode. Put your .tflite (+ .json) into the model folder.");
    }

    heap_mark("model + features + self-test");
#if CONFIG_KWS_INJECT_TEST
    if (s_model_ok) inject_loop();
#endif

    // 2) Microphone ------------------------------------------------------------------------------------------
    s_streaming_enabled = strlen(CONFIG_KWS_WIFI_SSID) > 0 && strlen(CONFIG_KWS_SERVER_URI) > 0;
    audio_input_config_t acfg = {
        .sck_gpio = CONFIG_KWS_MIC_SCK_GPIO,
        .ws_gpio = CONFIG_KWS_MIC_WS_GPIO,
        .sd_gpio = CONFIG_KWS_MIC_SD_GPIO,
        .gain_shift = CONFIG_KWS_MIC_GAIN_SHIFT,
        .ring = s_streaming_enabled,
    };
    audio_mic_check_t chk;
    err = audio_input_start(&acfg, &chk);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "microphone could not be started: %s", esp_err_to_name(err));
        status_led_set(40, 0, 0);
        return;
    }
    ESP_LOGI(TAG, "microphones: left (L/R to GND) %.1f dBFS%s, right (L/R to 3V3) %.1f dBFS%s, similarity %.2f -> %s",
             chk.level_dbfs[0], chk.use[0] ? "" : " (not used)", chk.level_dbfs[1], chk.use[1] ? "" : " (not used)",
             chk.correlation, chk.aligned ? "using both (time-aligned, mixed by their noise)"
                              : chk.use[0] && chk.use[1] ? "using both (mixed by their noise)" : chk.use[1] ? "using right" : "using left");
    s_mics_both = chk.use[0] && chk.use[1];
    s_mics_aligned = chk.aligned;
    if (!chk.ok) {
        ESP_LOGE(TAG, "NO DATA FROM THE MICROPHONES (raw 0x%08lx / 0x%08lx, %d / %d distinct values). Check: VDD->3V3, "
                      "GND->GND, L/R->GND (2nd mic: L/R->3V3), SCK->GPIO%d, WS->GPIO%d, SD->GPIO%d",
                 (unsigned long)chk.raw_first[0], (unsigned long)chk.raw_first[1], chk.distinct_values[0],
                 chk.distinct_values[1], CONFIG_KWS_MIC_SCK_GPIO, CONFIG_KWS_MIC_WS_GPIO, CONFIG_KWS_MIC_SD_GPIO);
        status_led_set(40, 0, 0);
    }
    // Audio + wake word task (core 1, away from Wi-Fi on core 0). Started now so no audio piles up in DMA.
    s_main_task = xTaskGetCurrentTaskHandle();
    xTaskCreatePinnedToCore(audio_task, "audio_kws", 3584, NULL, 10, NULL, 1);  // uses ~2.8 KB
    heap_mark("I2S + ring + audio task");
    if (s_model_ok) ESP_LOGI(TAG, "listening for \"%s\" ...", s_ms.wake_word);

    // 3) Wi-Fi + streaming ----------------------------------------------------------------------------------
    if (s_streaming_enabled) {
        wifi_start(CONFIG_KWS_WIFI_SSID, CONFIG_KWS_WIFI_PASSWORD);
        heap_mark("Wi-Fi driver + netif");
        streamer_config_t scfg = {
            .uri = CONFIG_KWS_SERVER_URI,
            .wake_word = s_ms.wake_word,
            .preroll_ms = CONFIG_KWS_PREROLL_MS,
            .min_ms = CONFIG_KWS_STREAM_MIN_MS,
            .max_ms = CONFIG_KWS_STREAM_MAX_MS,
            .silence_ms = CONFIG_KWS_STREAM_SILENCE_MS,
        };
        const esp_err_t serr = streamer_init(&scfg);
        heap_mark("WebSocket client + streamer");
        if (serr != ESP_OK) {
            ESP_LOGE(TAG, "streamer could not start");
            s_streaming_enabled = false;
        }
    } else {
        ESP_LOGI(TAG, "streaming is off (set the Wi-Fi name and server address in menuconfig to enable it)");
    }

    // 4) Button + status loop -------------------------------------------------------------------------------
    if (CONFIG_KWS_BUTTON_GPIO >= 0) {
        gpio_config_t io = {
            .pin_bit_mask = 1ULL << CONFIG_KWS_BUTTON_GPIO,
            .mode = GPIO_MODE_INPUT,
            .pull_up_en = GPIO_PULLUP_ENABLE,
        };
        gpio_config(&io);
    }
    const int64_t interval_us = (int64_t)CONFIG_KWS_STATUS_INTERVAL_S * 1000000;
    const int64_t tick_us = CONFIG_KWS_TELEMETRY_MS > 0 ? (int64_t)CONFIG_KWS_TELEMETRY_MS * 1000 : interval_us;
    int64_t last_tick = esp_timer_get_time(), next_tick = last_tick + tick_us, next_status = last_tick + interval_us;
    int pressed_ticks = 0;
    uint32_t n_status = 0;
    esp_task_wdt_add(NULL);  // (only here: the injection test loop above blocks on the serial port for good)
    for (;;) {
        esp_task_wdt_reset();
        // Wakes at once for a detection, else every 100 ms (button, LED, statistics)
        if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100))) {
            status_led_flash(0, 60, 0, 250);  // the model on the board decides: one green blink per detection
            ESP_LOGI(TAG, ">>> WAKE WORD \"%s\" DETECTED  (score %.2f, t = %.2f s)%s", s_ms.wake_word, s_det_score,
                     s_det_us / 1e6, s_det_extended ? " during a stream -> stream extended" : "");
        }
        float event_peak;
        int event_ms;
        bool event_detected;
        if (s_model_ok && ww_take_event(&event_peak, &event_ms, &event_detected))
            ESP_LOGI(TAG, "score event: peak %.2f over %d ms -> %s", event_peak, event_ms,
                     event_detected ? "detected" : "not detected (below threshold or in cool-down)");
        const int64_t now = esp_timer_get_time();
        status_led_poll();
        if (CONFIG_KWS_BUTTON_GPIO >= 0) {
            if (gpio_get_level(CONFIG_KWS_BUTTON_GPIO) == 0) {
                if (++pressed_ticks == 2) {  // held for 0.1-0.2 s
                    ESP_LOGI(TAG, "button pressed -> manual trigger");
                    if (s_streaming_enabled) {
                        streamer_trigger(1.0f, "button");
                    } else {
                        ESP_LOGW(TAG, "streaming is off: set the Wi-Fi name and server address in menuconfig");
                    }
                }
            } else {
                pressed_ticks = 0;
            }
        }
        if (now >= next_tick) {
            stats_tick(now - last_tick);
            last_tick = now;
            next_tick += tick_us;  // no drift; after a long stall restart the schedule instead of bursting
            if (next_tick <= now) next_tick = now + tick_us;
        }
        if (now >= next_status) {
            print_status();
            if (n_status++ % 6 == 0) print_tasks();  // at the first status line, then every minute
            next_status = now + interval_us;
        }
    }
}
