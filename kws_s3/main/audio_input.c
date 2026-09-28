#include "audio_input.h"

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"

static const char *TAG = "audio";

static i2s_chan_handle_t s_rx;
static float s_gain;
static float s_weight[2];  // share of the left / right microphone in the mix (from the boot check)

// 2nd-order Butterworth high-pass at 80 Hz (RBJ biquad). Removes the DC offset and the large infrasonic drift the
// INMP441 picks up (it otherwise eats the headroom and clips speech). Voice is not affected.
static float s_b0, s_b1, s_b2, s_a1, s_a2;
static float s_x1, s_x2, s_y1, s_y2;

// Ring of the last AUDIO_RING_SAMPLES samples (mu-law). One writer (audio_input_read), one reader (the streamer).
// s_wpos counts samples since boot; the power-of-2 size keeps "pos & mask" correct when it wraps after 74 h.
static uint8_t *s_ring;
static atomic_uint s_wpos;
#define RING_MASK (AUDIO_RING_SAMPLES - 1)

// Statistics (protected by a spinlock because they are read from another core)
static portMUX_TYPE s_stats_lock = portMUX_INITIALIZER_UNLOCKED;
static int64_t s_sumsq;
static uint32_t s_nsamples;
static float s_peak_ms;  // highest block mean-square
static uint32_t s_net_drop_samples;
static uint64_t s_busy_us;
static float s_noise_floor = 50.0f;
static volatile uint32_t s_dma_overflows;  // written from the I2S interrupt

static bool IRAM_ATTR on_rx_overflow(i2s_chan_handle_t handle, i2s_event_data_t *event, void *ctx) {
    s_dma_overflows++;
    return false;
}

static inline int16_t sat16(int32_t v) { return v > 32767 ? 32767 : (v < -32768 ? -32768 : (int16_t)v); }

static float ms_to_dbfs(double mean_square) {
    if (mean_square <= 0) return -120.0f;
    return (float)(10.0 * log10(mean_square) - 20.0 * log10(32768.0));
}

static esp_err_t i2s_setup(const audio_input_config_t *cfg) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    // 6 x 20 ms = 120 ms: rides out a flash write (the task stalls while the flash cache is off). Wi-Fi no longer
    // writes its settings to flash (WIFI_STORAGE_RAM); "audio lost: i2s" in the status line counts any overflow.
    chan_cfg.dma_desc_num = 6;
    chan_cfg.dma_frame_num = AUDIO_BLOCK_SAMPLES;
    esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &s_rx);
    if (err != ESP_OK) return err;

    // The INMP441 sends 24-bit samples inside 32-bit I2S slots (Philips format, 64 clocks per frame). Both slots
    // are read: two microphones share SCK, WS and SD; the one with L/R to GND answers in the left slot, the one
    // with L/R to 3V3 in the right slot (each drives SD only during its own slot).
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)cfg->sck_gpio,
            .ws = (gpio_num_t)cfg->ws_gpio,
            .dout = I2S_GPIO_UNUSED,
            .din = (gpio_num_t)cfg->sd_gpio,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };

    err = i2s_channel_init_std_mode(s_rx, &std_cfg);
    if (err != ESP_OK) return err;
    i2s_event_callbacks_t cbs = {.on_recv_q_ovf = on_rx_overflow};
    i2s_channel_register_event_callback(s_rx, &cbs, NULL);
    return i2s_channel_enable(s_rx);
}

// Reads ~0.3 s from both slots and decides which microphones to use. A slot counts as a working microphone if
// its samples vary (not all zeros / all ones / a constant) at a plausible level (an empty slot on a floating SD line
// reads garbage near full scale). Two working microphones within 10 dB of each other are averaged.
static void mic_check(audio_mic_check_t *out) {
    static int32_t raw[AUDIO_BLOCK_SAMPLES * 2];
    size_t bytes = 0;
    double sum[2] = {0, 0}, sumsq[2] = {0, 0}, cross = 0;
    int32_t prev[2] = {0, 0};  // the similarity uses sample-to-sample differences (a strong high-pass): the power-up
    double dss[2] = {0, 0};    // drift of the two INMP441s is shared and would make any pair look identical
    uint32_t n = 0;
    int32_t seen[2][8];
    int nseen[2] = {0, 0};
    memset(out, 0, sizeof(*out));
    for (int block = 0; block < 25; block++) {  // 25 x 20 ms
        if (i2s_channel_read(s_rx, raw, sizeof(raw), &bytes, pdMS_TO_TICKS(500)) != ESP_OK) continue;
        if (block < 10) continue;                   // the INMP441 needs ~85 ms to start; skip 200 ms
        const size_t frames = bytes / (2 * sizeof(int32_t));
        if (block == 10 && frames) {
            out->raw_first[0] = (uint32_t)raw[0];
            out->raw_first[1] = (uint32_t)raw[1];
        }
        for (size_t i = 0; i < frames; i++) {
            for (int c = 0; c < 2; c++) {
                const int32_t v = raw[2 * i + c], s = v >> 16;
                sum[c] += s;
                sumsq[c] += (double)s * s;
                if (n == 0) prev[c] = s;  // no difference for the very first sample
                const int32_t d = s - prev[c];
                prev[c] = s;
                if (c == 0 && n) cross += (double)d * ((raw[2 * i + 1] >> 16) - prev[1]);
                dss[c] += (double)d * d;
                bool found = false;
                for (int k = 0; k < nseen[c]; k++) found |= (seen[c][k] == v);
                if (!found && nseen[c] < 8) seen[c][nseen[c]++] = v;
            }
            n++;
        }
    }
    // Levels and similarity without the DC offset (the INMP441's offset would dominate both)
    double var[2] = {0, 0};
    bool live[2];
    for (int c = 0; c < 2; c++) {
        var[c] = n ? sumsq[c] / n - (sum[c] / n) * (sum[c] / n) : 0;
        out->distinct_values[c] = nseen[c];
        out->level_dbfs[c] = ms_to_dbfs(var[c]);
        live[c] = nseen[c] >= 3 && out->level_dbfs[c] > -100.0f && out->level_dbfs[c] < -6.0f;
    }
    out->correlation = (dss[0] > 0 && dss[1] > 0) ? (float)(cross / sqrt(dss[0] * dss[1])) : 0.0f;
    if (live[0] && live[1] && fabsf(out->level_dbfs[0] - out->level_dbfs[1]) < 10.0f) {
        out->use[0] = out->use[1] = true;
    } else if (live[0] || live[1]) {
        const int c = (live[0] && (!live[1] || out->level_dbfs[0] >= out->level_dbfs[1])) ? 0 : 1;
        out->use[c] = true;
    } else {
        out->use[0] = true;  // nothing works: keep listening on the left slot (the wiring check prints an error)
    }
    out->ok = live[0] || live[1];
}

size_t audio_input_read(int16_t pcm[AUDIO_BLOCK_SAMPLES]) {
    static int32_t raw[AUDIO_BLOCK_SAMPLES * 2];  // left, right, left, right, ...
    size_t bytes = 0;
    if (i2s_channel_read(s_rx, raw, sizeof(raw), &bytes, pdMS_TO_TICKS(1000)) != ESP_OK || bytes == 0) {
        ESP_LOGW(TAG, "no data from I2S");
        return 0;
    }
    const int64_t t0 = esp_timer_get_time();
    const size_t n = bytes / (2 * sizeof(int32_t));
    int64_t sumsq = 0;  // integer: the S3 has no double-precision FPU
    for (size_t i = 0; i < n; i++) {
        // 24-bit samples, mixed (both microphones: their average) and filtered before the gain so nothing clips
        const float x = s_weight[0] * (float)(raw[2 * i] >> 8) + s_weight[1] * (float)(raw[2 * i + 1] >> 8);
        const float y = s_b0 * x + s_b1 * s_x1 + s_b2 * s_x2 - s_a1 * s_y1 - s_a2 * s_y2;
        s_x2 = s_x1;
        s_x1 = x;
        s_y2 = s_y1;
        s_y1 = y;
        const float v = y * s_gain;
        pcm[i] = sat16((int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f));  // one FPU instruction; lrintf is a library call
        sumsq += (int32_t)pcm[i] * pcm[i];
    }

    if (s_ring) {
        const uint32_t w = atomic_load(&s_wpos);
        for (size_t i = 0; i < n; i++) s_ring[(w + i) & RING_MASK] = mulaw_encode(pcm[i]);
        atomic_store(&s_wpos, w + (uint32_t)n);  // publish after the data is written
    }

    // statistics + noise floor (fast down, slow up)
    const float ms = n ? (float)sumsq / (float)n : 0.0f, rms = sqrtf(ms);
    const int64_t busy = esp_timer_get_time() - t0;
    portENTER_CRITICAL(&s_stats_lock);
    s_sumsq += sumsq;
    s_nsamples += n;
    if (ms > s_peak_ms) s_peak_ms = ms;
    s_busy_us += (uint64_t)busy;
    s_noise_floor = (rms < s_noise_floor) ? (0.9f * s_noise_floor + 0.1f * rms) : (s_noise_floor * 1.002f + 0.01f);
    if (s_noise_floor < 1.0f) s_noise_floor = 1.0f;
    portEXIT_CRITICAL(&s_stats_lock);
    return n;
}

uint32_t audio_ring_pos(void) { return atomic_load(&s_wpos); }

size_t audio_ring_read(uint32_t *pos, uint8_t *dst, size_t max) {
    if (!s_ring) return 0;
    // Keep one block of margin from the oldest sample: the writer overwrites it while filling the next block.
    const uint32_t keep = AUDIO_RING_SAMPLES - AUDIO_BLOCK_SAMPLES;
    uint32_t w = atomic_load(&s_wpos);
    if (w - *pos > keep) {  // too far behind: skip to the oldest audio still in the ring
        const uint32_t skip = (w - keep) - *pos;
        portENTER_CRITICAL(&s_stats_lock);
        s_net_drop_samples += skip;
        portEXIT_CRITICAL(&s_stats_lock);
        *pos = w - keep;
    }
    size_t n = w - *pos;
    if (n > max) n = max;
    const uint32_t idx = *pos & RING_MASK;
    const size_t first = n < AUDIO_RING_SAMPLES - idx ? n : AUDIO_RING_SAMPLES - idx;
    memcpy(dst, s_ring + idx, first);
    memcpy(dst + first, s_ring, n - first);
    *pos += (uint32_t)n;
    return n;
}

esp_err_t audio_input_start(const audio_input_config_t *cfg, audio_mic_check_t *check_out) {
    s_gain = (float)(1 << cfg->gain_shift) / 256.0f;  // 24-bit sample -> 16-bit, with digital gain
    const float w0 = 2.0f * (float)M_PI * 80.0f / AUDIO_SAMPLE_RATE, alpha = sinf(w0) / (2.0f * 0.7071f), c = cosf(w0);
    const float a0 = 1.0f + alpha;
    s_b0 = (1.0f + c) / 2.0f / a0;
    s_b1 = -(1.0f + c) / a0;
    s_b2 = s_b0;
    s_a1 = -2.0f * c / a0;
    s_a2 = (1.0f - alpha) / a0;

    if (cfg->ring) {
        s_ring = heap_caps_calloc(AUDIO_RING_SAMPLES, 1, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!s_ring) return ESP_ERR_NO_MEM;
    }

    esp_err_t err = i2s_setup(cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2S setup failed: %s (check the GPIO numbers in menuconfig)", esp_err_to_name(err));
        return err;
    }
    ESP_LOGI(TAG, "I2S running: 16 kHz, 32-bit slots, both channels, SCK=%d WS=%d SD=%d, gain x%d", cfg->sck_gpio,
             cfg->ws_gpio, cfg->sd_gpio, 1 << cfg->gain_shift);

    audio_mic_check_t chk;
    mic_check(&chk);
    const float share = (chk.use[0] && chk.use[1]) ? 0.5f : 1.0f;
    s_weight[0] = chk.use[0] ? share : 0.0f;
    s_weight[1] = chk.use[1] ? share : 0.0f;
    if (check_out) *check_out = chk;
    return ESP_OK;
}

void audio_input_take_stats(audio_stats_t *out) {
    portENTER_CRITICAL(&s_stats_lock);
    const int64_t sumsq = s_sumsq;
    const float peak_ms = s_peak_ms;
    const uint32_t n = s_nsamples;
    out->net_drops = (s_net_drop_samples + AUDIO_BLOCK_SAMPLES - 1) / AUDIO_BLOCK_SAMPLES;
    out->busy_us = s_busy_us;
    out->dma_overflows = s_dma_overflows;
    s_sumsq = 0;
    s_nsamples = 0;
    s_peak_ms = 0;
    s_net_drop_samples = 0;
    s_busy_us = 0;
    s_dma_overflows = 0;
    portEXIT_CRITICAL(&s_stats_lock);
    out->level_dbfs = n ? ms_to_dbfs((double)sumsq / n) : -120.0f;
    out->peak_dbfs = ms_to_dbfs(peak_ms);
}

float audio_input_noise_floor(void) {
    portENTER_CRITICAL(&s_stats_lock);
    float nf = s_noise_floor;
    portEXIT_CRITICAL(&s_stats_lock);
    return nf;
}
