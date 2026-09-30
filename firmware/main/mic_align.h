// Time alignment of the two microphones before they are averaged (delay-and-sum).
//
// Both INMP441 share SCK and WS, so their samples are taken at the same instants (one I2S frame = one left + one right
// sample, no drift). What differs is when the sound gets there: a talker off to one side reaches the nearer microphone
// up to spacing / 343 m/s earlier (55 mm: 160 us = 2.6 samples at 16 kHz). A plain average then cancels part of the
// voice (for a talker in line with the pair: a notch at 3.1 kHz, -6 dB already at 2 kHz).
//
// mic_align measures that delay while someone speaks (cross-correlation of the two microphones, peak refined to a
// fraction of a sample) and delays the earlier microphone by it (4-tap Lagrange fractional delay) before averaging,
// so the voice adds up in phase from any direction and each microphone's own noise still averages down (+3 dB SNR).
// The mix follows the later microphone by one sample (62.5 us, the fractional-delay filter's look-ahead).
//
// Each microphone's share of the mix follows its measured background noise (inverse noise power, the best mix for two
// microphones that hear the voice equally loud): 50/50 for a healthy pair, while a microphone with a wiring or supply
// problem that adds 20 dB of noise drops to 1 % instead of drowning the good one.
//
// Pure C, no ESP-IDF: test/test_mic_align.c builds it on the PC.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIC_ALIGN_MAX_K 8  // correlation lags searched each way at most (spacing up to ~120 mm at 16 kHz)
#define MIC_ALIGN_HIST (MIC_ALIGN_MAX_K + 2)  // frames of the previous block the filter and the estimate reach back

typedef struct {
    int32_t hist[2 * MIC_ALIGN_HIST];  // the previous block's last frames (left, right, left, ...)
    float c[2 * MIC_ALIGN_MAX_K + 1], e[2];  // this block: correlation at lags -K..K, energy of each microphone
    float nf[2];     // background of e[] per microphone (follows the quietest blocks)
    float w[2];      // share of each microphone in the mix (left + right = 1)
    float lag;       // how many samples the right microphone hears the talker after the left one (< 0: before)
    float limit;     // largest possible |lag|: spacing / speed of sound, + half a sample of slack
    float h[4];      // fractional-delay taps for the earlier microphone
    int K;           // lags searched each way
    int k;           // whole samples of the earlier microphone's delay (the later one is delayed by 1)
    int lead;        // the earlier microphone: 0 = left, 1 = right
    bool both;      // both microphones in use: shares follow their noise
    bool estimate;   // both in use and the spacing is known: time alignment
    bool correlate;  // correlate the current block (after every 4th speech block)
    uint8_t speech_blocks;
    uint32_t updates;  // delay estimates taken since boot
} mic_align_t;

// Weights: which microphones are used (> 0; the shares then follow the noise). spacing_mm = 0: no time alignment.
// One weight 0 or spacing 0: a plain weighted mix, one sample late.
void mic_align_init(mic_align_t *a, float spacing_mm, float sample_rate, float w_left, float w_right);

// raw: 2 * MIC_ALIGN_HIST int32 of room (for the history), then one I2S block of `frames` (>= MIC_ALIGN_HIST)
// interleaved left/right 32-bit slots (24-bit samples). Writes the frames mixed, time-aligned samples in 24-bit units
// to raw[0 .. frames-1].
void mic_align_mix(mic_align_t *a, int32_t *raw, size_t frames);

// Call after each block. Updates both microphones' backgrounds and shares. speech = the block was well above the
// background: then the block's correlation refines the delay (if both microphones heard the same sound). Quiet blocks
// leave the delay alone and cost no correlation.
void mic_align_update(mic_align_t *a, bool speech);

#ifdef __cplusplus
}
#endif
