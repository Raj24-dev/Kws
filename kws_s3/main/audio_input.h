// Microphone input: reads one or two INMP441 over I2S (hardware + DMA), mixes them to 16 kHz 16-bit mono PCM and
// keeps the last ~2 s in a small ring buffer (8-bit mu-law) that the network streamer reads from.
// Two microphones (e.g. 55 mm apart) are averaged: a talker in front of the pair (perpendicular to the line between
// the microphones) reaches both at the same time, so speech adds up while each microphone's own noise does not
// (+3 dB SNR). A talker along that line loses a little around 3 kHz (the 0.16 ms path difference).
//
// There is no capture task: the wake word task calls audio_input_read() in a loop, so capture and detection run
// in one task (the I2S DMA buffer is the queue between the hardware and the task).
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "sdkconfig.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_SAMPLE_RATE 16000
#define AUDIO_BLOCK_SAMPLES 320     // 20 ms per block
// History for the streamer (pre-roll + network slack), 1 byte per sample, power of 2: 8 KB = 512 ms for pre-rolls up to
// 250 ms; bigger only for long pre-rolls (e.g. KWS_PREROLL_MS=1000 to collect training recordings of the wake word)
#define AUDIO_RING_SAMPLES (CONFIG_KWS_PREROLL_MS <= 250 ? 8192 : CONFIG_KWS_PREROLL_MS <= 750 ? 16384 : 32768)

typedef struct {
    int sck_gpio;
    int ws_gpio;
    int sd_gpio;
    int gain_shift;       // 0..4  (x1..x16)
    bool ring;            // keep the audio history for streaming (false = no Wi-Fi streaming, saves 32 KB)
} audio_input_config_t;

typedef struct {           // [0] = left slot (L/R pin to GND), [1] = right slot (L/R pin to 3V3)
    bool ok;                  // false = no data from any microphone (wiring problem)
    bool use[2];              // microphones in use (both = averaged)
    float level_dbfs[2];      // loudness during the check
    uint32_t raw_first[2];    // first raw 32-bit word (helps diagnose wiring)
    int distinct_values[2];   // how many different sample values were seen
    float correlation;        // similarity of the two slots (-1..1); ~1 for two microphones side by side at low
                              // frequencies, ~0 if one slot is noise
} audio_mic_check_t;

// Sets up I2S and runs a short microphone check (which slots have a working microphone).
esp_err_t audio_input_start(const audio_input_config_t *cfg, audio_mic_check_t *check_out);

// Waits for the next 20 ms block (reads I2S, filters, stores it in the ring). Returns the number of samples
// (AUDIO_BLOCK_SAMPLES), or 0 if the microphone sent nothing for 1 s.
size_t audio_input_read(int16_t pcm[AUDIO_BLOCK_SAMPLES]);

// Ring buffer of the recent audio, 1 byte (mu-law) per sample. Positions count samples since boot.
uint32_t audio_ring_pos(void);  // position of the next sample to be written
// Copies up to max samples from *pos into dst and advances *pos. A reader that fell more than the ring behind
// jumps to the oldest audio still there; the skipped audio is counted in audio_stats_t.net_drops.
size_t audio_ring_read(uint32_t *pos, uint8_t *dst, size_t max);

// Statistics for the status line (values since the previous call)
typedef struct {
    float level_dbfs;       // average loudness
    float peak_dbfs;        // loudest block
    uint32_t net_drops;     // 20 ms blocks the streamer lost because the network was too slow
    uint64_t busy_us;       // time spent capturing and filtering audio
    uint32_t dma_overflows; // I2S buffers lost because the task did not read them in time
} audio_stats_t;
void audio_input_take_stats(audio_stats_t *out);

// Current background-noise estimate (RMS in 16-bit units), used to detect silence while streaming
float audio_input_noise_floor(void);

// G.711 mu-law: 16-bit PCM <-> 8 bits per sample (telephone quality, ~38 dB SNR, fine for speech recognition)
static inline uint8_t mulaw_encode(int16_t pcm) {
    int x = pcm;
    const int sign = x < 0 ? 0x80 : 0;
    if (sign) x = -x;
    if (x > 32635) x = 32635;
    x += 0x84;
    const int exp = (31 - __builtin_clz((unsigned)x)) - 7;  // 0..7
    return (uint8_t) ~(sign | (exp << 4) | ((x >> (exp + 3)) & 0x0F));
}
static inline int16_t mulaw_decode(uint8_t u) {
    u = (uint8_t)~u;
    const int t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4);
    return (int16_t)((u & 0x80) ? (0x84 - t) : (t - 0x84));
}

#ifdef __cplusplus
}
#endif
