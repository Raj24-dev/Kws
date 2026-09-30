// Host test for main/mic_align.c (no board needed):
//   gcc -O2 -std=c99 -I../main test_mic_align.c ../main/mic_align.c -lm -o test_mic_align && ./test_mic_align
// A broadband "voice" (40 random tones, 150-6500 Hz) reaches the right microphone a known fraction of a sample after
// the left one; the estimate must find that delay and the aligned mix must keep the voice a plain average cancels.
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mic_align.h"

#define FS 16000.0
#define N 320
#define PI 3.14159265358979
#define TONES 40

static double f_[TONES], ph_[TONES];

static double voice(double t) {  // t in samples
    double s = 0;
    for (int j = 0; j < TONES; j++) s += sin(2 * PI * f_[j] * t / FS + ph_[j]);
    return s / sqrt(TONES / 2.0);  // RMS 1
}
static double noise(void) { return (rand() / (double)RAND_MAX - 0.5) * 3.46; }  // RMS 1

// Feeds blocks: left(t) = src(t), right(t) = src(t - lag), plus independent noise per microphone (noise_rms; the
// right one right_noise_x times more). Returns the RMS of the mix over the last half of the blocks, relative to the
// source's.
static long t0;
static double right_noise_x = 1.0;
static int32_t raw[2 * (MIC_ALIGN_HIST + N)], *const blk = raw + 2 * MIC_ALIGN_HIST;  // the layout audio_input.c uses
static double run(mic_align_t *a, double (*src)(double), double lag, double noise_rms, int blocks, bool speech) {
    double out2 = 0, in2 = 0;
    const double amp = 1 << 16;  // -42 dBFS in 24-bit units: room for the 34 dB louder noise of test 3
    for (int b = 0; b < blocks; b++, t0 += N) {
        for (int i = 0; i < N; i++) {
            const double l = src(t0 + i) + noise_rms * noise(), r = src(t0 + i - lag) + right_noise_x * noise_rms * noise();
            blk[2 * i] = (int32_t)(l * amp) * 256;
            blk[2 * i + 1] = (int32_t)(r * amp) * 256;
        }
        mic_align_mix(a, raw, N);
        mic_align_update(a, speech);
        if (b >= blocks / 2)
            for (int i = 0; i < N; i++) {
                out2 += (double)raw[i] * raw[i];
                const double s = src(t0 + i) * amp;
                in2 += s * s;
            }
    }
    return sqrt(out2 / in2);
}

static double silence(double t) { return 0 * t; }
static double tone3k(double t) { return sqrt(2.0) * sin(2 * PI * 3000 * t / FS); }

int main(void) {
    srand(1);
    for (int j = 0; j < TONES; j++) {
        f_[j] = 150 + 6350.0 * rand() / (double)RAND_MAX;
        ph_[j] = 2 * PI * rand() / (double)RAND_MAX;
    }
    mic_align_t a;

    // 1) The delay is found for talkers anywhere around a 55 mm pair (|lag| <= 2.57 samples), fractions included
    const double lags[] = {-2.5, -1.3, 0.0, 0.6, 2.2, 2.55};
    for (int i = 0; i < 6; i++) {
        mic_align_init(&a, 55, FS, 0.5f, 0.5f);
        run(&a, voice, lags[i], 0.03, 40, true);  // 0.8 s of speech
        printf("true lag %+.2f -> estimated %+.3f (%u updates)\n", lags[i], a.lag, (unsigned)a.updates);
        assert(fabs(a.lag - lags[i]) < 0.15);
    }

    // 2) Talker in line with the pair (lag 2.5): a 3 kHz component survives the aligned mix, a plain average cancels it
    mic_align_init(&a, 55, FS, 0.5f, 0.5f);
    run(&a, voice, 2.5, 0.03, 20, true);
    const double aligned = run(&a, tone3k, 2.5, 0.0, 10, false);  // speech=false: keep the delay found on the voice
    mic_align_init(&a, 0, FS, 0.5f, 0.5f);  // spacing 0 = plain average
    const double plain = run(&a, tone3k, 2.5, 0.0, 10, false);
    printf("3 kHz at lag 2.5: aligned mix %.3f, plain average %.3f (of the source level)\n", aligned, plain);
    assert(aligned > 0.9 && plain < 0.15);

    // 3) Sound the two microphones do not share (independent noise) does not move a delay already found
    mic_align_init(&a, 55, FS, 0.5f, 0.5f);
    run(&a, voice, 1.0, 0.03, 20, true);
    const float before = a.lag;
    const uint32_t n_before = a.updates;
    run(&a, voice, 1.0, 50.0, 30, true);  // voice 34 dB below the uncorrelated noise
    printf("uncorrelated noise: lag %+.3f -> %+.3f, %u updates\n", before, a.lag, (unsigned)(a.updates - n_before));
    assert(fabs(a.lag - before) < 0.2);

    // 4) One microphone (the other's weight 0): its samples pass unchanged, one sample late
    mic_align_init(&a, 55, FS, 1.0f, 0.0f);
    int32_t prev = 0;
    for (int i = 0; i < N; i++) blk[2 * i] = (i * 7919 % 20000 - 10000) * 256, blk[2 * i + 1] = 12345 * 256;
    int32_t left[N];
    for (int i = 0; i < N; i++) left[i] = blk[2 * i] >> 8;
    mic_align_mix(&a, raw, N);
    for (int i = 0; i < N; i++) {
        assert(raw[i] == prev);
        prev = left[i];
    }
    printf("single microphone: exact copy, 1 sample late\n");

    // 5) Shares follow the background noise: 50/50 for a healthy pair, ~1 % for a microphone 20 dB noisier
    mic_align_init(&a, 55, FS, 0.5f, 0.5f);
    run(&a, silence, 0.0, 0.03, 50, false);
    printf("equal noise: left share %.3f\n", a.w[0]);
    assert(fabs(a.w[0] - 0.5) < 0.05);
    right_noise_x = 10.0;  // e.g. a wire comes loose: the background rises +0.9 dB/s, so 20 dB takes ~23 s
    run(&a, silence, 0.0, 0.03, 1300, false);
    printf("right 20 dB noisier (26 s later): right share %.4f\n", a.w[1]);
    assert(a.w[1] < 0.02 && fabs(a.w[0] + a.w[1] - 1.0) < 1e-5);
    right_noise_x = 1.0;
    printf("all mic_align tests passed\n");
    return 0;
}
