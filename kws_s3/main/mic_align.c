#include "mic_align.h"

#include <math.h>
#include <string.h>

#define H MIC_ALIGN_HIST
// Sample n of microphone c (0 = left, 1 = right) as the 32-bit slot value (the 24-bit sample x 256: the scale is folded
// into the mix weights instead of a shift per sample); x = raw + 2 * H, n >= -H
#define S(n, c) ((float)x[2 * (n) + (c)])
// Its sample-to-sample difference (a whitened signal: a sharp correlation peak instead of the broad one of voice)
#define D(n, c) (S(n, c) - S((n) - 1, c))

// Earlier microphone delayed by 1 + |lag| samples (k whole + m fraction), the later one by exactly 1: the Lagrange
// filter reads one sample ahead of k, so 1 is the least delay that keeps it causal.
static void set_delay(mic_align_t *a) {
    const float delay = 1.0f + fabsf(a->lag);
    a->k = (int)delay;
    const float m = delay - (float)a->k;
    a->h[0] = -m * (m - 1.0f) * (m - 2.0f) / 6.0f;          // x[n-k+1]
    a->h[1] = (m + 1.0f) * (m - 1.0f) * (m - 2.0f) / 2.0f;  // x[n-k]
    a->h[2] = -(m + 1.0f) * m * (m - 2.0f) / 2.0f;          // x[n-k-1]
    a->h[3] = (m + 1.0f) * m * (m - 1.0f) / 6.0f;           // x[n-k-2]
    a->lead = a->lag < 0.0f;
}

void mic_align_init(mic_align_t *a, float spacing_mm, float sample_rate, float w_left, float w_right) {
    memset(a, 0, sizeof(*a));
    a->w[0] = w_left;
    a->w[1] = w_right;
    const float max_lag = spacing_mm * 1e-3f / 343.0f * sample_rate;
    // The peak needs a neighbour on each side for the fraction: search 2 lags past the physical limit
    a->K = (int)max_lag + 2;
    if (a->K > MIC_ALIGN_MAX_K) a->K = MIC_ALIGN_MAX_K;
    a->limit = fminf(max_lag + 0.5f, (float)(a->K - 1));  // keeps k <= K, so the filter stays inside the history
    a->both = w_left > 0.0f && w_right > 0.0f;
    a->estimate = a->both && spacing_mm > 0.0f;
    set_delay(a);
}

void mic_align_mix(mic_align_t *a, int32_t *raw, size_t frames) {
    memcpy(raw, a->hist, sizeof(a->hist));  // the previous block's last frames, right before this one
    const int32_t *x = raw + 2 * H;
    const int N = (int)frames, K = a->K;

    // 1) Background of each microphone and (if asked) the correlation for the delay, every 4th sample: plenty for
    //    an estimate at a quarter of the cost. c[K + j] = sum of right[n] * left[n - j]: peaks at j = lag.
    float e0 = 0.0f, e1 = 0.0f, c[2 * MIC_ALIGN_MAX_K + 1] = {0};
    for (int n = 1; n + K < N; n += 4) {
        const float dl = D(n, 0), dr = D(n, 1);
        e0 += dl * dl;
        e1 += dr * dr;
        if (a->correlate)
            for (int j = -K; j <= K; j++) c[K + j] += dr * D(n - j, 0);
    }
    a->e[0] = e0;
    a->e[1] = e1;
    memcpy(a->c, c, sizeof(c));

    // 2) Mix, in place: raw[n] lies below every index a later sample reads (>= 2n)
    const int e = a->lead, k = a->k;
    const float we = a->w[e] / 256.0f, wl = a->w[!e] / 256.0f;  // 32-bit slot -> 24-bit units
    const float g0 = we * a->h[0], g1 = we * a->h[1], g2 = we * a->h[2], g3 = we * a->h[3];
#if __GNUC__ >= 8
#pragma GCC unroll 2  // two samples per pass: the in-order FPU overlaps them (latency, not instructions, is the cost)
#endif
    for (int n = 0; n < N; n++) {
        const float s1 = g0 * S(n - k + 1, e) + g1 * S(n - k, e), s2 = g2 * S(n - k - 1, e) + g3 * S(n - k - 2, e);
        raw[n] = (int32_t)(s1 + s2 + wl * S(n - 1, !e));  // two partial sums: a shorter chain of dependent adds
    }

    // 3) This block's last frames are the next one's history (above index N: not overwritten by the mix)
    memcpy(a->hist, x + 2 * (N - H), sizeof(a->hist));
}

void mic_align_update(mic_align_t *a, bool speech) {
    if (a->both) {
        // Background per microphone: fast down, slow up (+0.9 dB/s), like the level's noise floor in audio_input.c.
        // Shares = inverse noise power. ponytail: assumes both hear the voice equally loud (INMP441: +-1 dB); a
        // microphone that is nearly deaf but quiet would win, the boot check only rejects fully dead ones.
        for (int c = 0; c < 2; c++)
            a->nf[c] = (a->nf[c] <= 0.0f) ? a->e[c] : a->e[c] < a->nf[c] ? 0.9f * a->nf[c] + 0.1f * a->e[c] : a->nf[c] * 1.004f;
        const float sum = a->nf[0] + a->nf[1];
        if (sum > 0.0f) {
            a->w[0] = a->nf[1] / sum;
            a->w[1] = a->nf[0] / sum;
        }
    }
    if (a->correlate && speech) {
        const int K = a->K;
        int best = -K;
        for (int j = -K + 1; j <= K; j++)
            if (a->c[K + j] > a->c[K + best]) best = j;
        const float peak = a->c[K + best], norm = sqrtf(a->e[0] * a->e[1]);
        // Only a sound both microphones heard alike (normalised correlation >= 0.4, e.g. not wind or handling noise)
        // with its peak inside the physically possible range
        if (best > -K && best < K && norm > 0.0f && peak >= 0.4f * norm) {
            // Fraction from a cosine through the peak and its neighbours (the correlation of band-limited sound is
            // cosine-shaped there; a parabola pulls the result towards whole samples by up to 0.15)
            const float y0 = a->c[K + best - 1], y2 = a->c[K + best + 1];
            const float w = acosf(fmaxf(-1.0f, fminf(1.0f, (y0 + y2) / (2.0f * peak))));
            float lag = (float)best + (w > 1e-3f ? atanf((y2 - y0) / (2.0f * peak * sinf(w))) / w : 0.0f);
            lag = fmaxf(-a->limit, fminf(a->limit, lag));
            a->lag += (a->updates ? 0.5f : 1.0f) * (lag - a->lag);  // the first one as is, then smooth over two
            set_delay(a);
            a->updates++;
        }
    }
    // Correlate the block after every 4th speech block (an estimate per 80 ms of speech: the talker moves slowly, and
    // the CPU budget is tight)
    a->correlate = a->estimate && speech && (++a->speech_blocks & 3) == 0;
}
