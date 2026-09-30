#include "wav_selftest.h"

#include <stdio.h>
#include <string.h>

#include "wake_word.h"

// Must match training/check_model.py so both give the same numbers.
#define ST_RATE 16000
#define ST_LEAD_MS 1200     // quiet noise before the word (the detector ignores the first second)
#define ST_TAIL_MS 600      // quiet noise after the word
#define ST_NEG_MS 3000      // length of the negative (noise only) test
#define ST_CHUNK 320        // feed 20 ms at a time, like the live microphone

static uint32_t s_lcg;
static int16_t noise_sample(void) {  // deterministic low-level noise, about -66 dBFS
    s_lcg = s_lcg * 1664525u + 1013904223u;
    return (int16_t)((int32_t)((s_lcg >> 16) % 61u) - 30);
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

// Feeds samples through the engine and records the first detection time.
static void feed(const int16_t *s, size_t n, size_t *fed, selftest_result_t *r, size_t clip_start) {
    for (size_t off = 0; off < n; off += ST_CHUNK) {
        size_t k = (n - off < ST_CHUNK) ? n - off : ST_CHUNK;
        bool det = ww_process(s + off, k, NULL);
        *fed += k;
        if (det && r->detect_ms < 0) {
            r->detected = true;
            r->detect_ms = (int)(((long)*fed - (long)clip_start) * 1000 / ST_RATE);
        }
    }
}

// Highest averaged probability since the previous call
static float take_max_score(void) {
    ww_stats_t st;
    ww_take_stats(&st);
    return st.max_avg_probability;
}

static void feed_noise(int ms, size_t *fed, selftest_result_t *r, size_t clip_start) {
    int16_t buf[ST_CHUNK];
    size_t total = (size_t)ms * ST_RATE / 1000;
    while (total) {
        size_t k = total < ST_CHUNK ? total : ST_CHUNK;
        for (size_t i = 0; i < k; i++) buf[i] = noise_sample();
        feed(buf, k, fed, r, clip_start);
        total -= k;
    }
}

void wav_selftest_run(const uint8_t *wav, size_t len, selftest_result_t *pos, selftest_result_t *neg) {
    memset(pos, 0, sizeof(*pos));
    memset(neg, 0, sizeof(*neg));
    pos->detect_ms = neg->detect_ms = -1;

    // ---- negative test: noise only, must stay silent ----
    ww_reset();
    take_max_score();
    s_lcg = 2;
    size_t fed = 0;
    neg->ran = true;
    feed_noise(ST_NEG_MS, &fed, neg, 0);
    neg->max_score = take_max_score();

    // ---- positive test: parse the WAV file ----
    if (!wav || len < 44 || memcmp(wav, "RIFF", 4) || memcmp(wav + 8, "WAVE", 4)) {
        snprintf(pos->error, sizeof(pos->error), "no WAV file in the model folder (optional)");
        ww_reset();
        return;
    }
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t *data = NULL;
    uint32_t data_len = 0;
    size_t p = 12;
    while (p + 8 <= len) {
        uint32_t csize = rd32(wav + p + 4);
        const uint8_t *c = wav + p + 8;
        if (!memcmp(wav + p, "fmt ", 4) && csize >= 16) {
            fmt = rd16(c);
            ch = rd16(c + 2);
            rate = rd32(c + 4);
            bits = rd16(c + 14);
        } else if (!memcmp(wav + p, "data", 4)) {
            data = c;
            data_len = csize;
            if (data + data_len > wav + len) data_len = (uint32_t)(wav + len - data);
            break;
        }
        p += 8 + csize + (csize & 1);
    }
    if (!data || (fmt != 1 && fmt != 0xFFFE) || bits != 16 || (ch != 1 && ch != 2) || rate != ST_RATE) {
        snprintf(pos->error, sizeof(pos->error), "WAV must be 16 kHz, 16-bit PCM (got %u Hz, %u-bit, %u ch)",
                 (unsigned)rate, bits, ch);
        ww_reset();
        return;
    }

    ww_reset();
    take_max_score();
    s_lcg = 1;
    fed = 0;
    pos->ran = true;
    feed_noise(ST_LEAD_MS, &fed, pos, 0);
    const size_t clip_start = fed;
    const size_t frames = data_len / (2u * ch);
    pos->clip_ms = (int)(frames * 1000 / ST_RATE);
    int16_t buf[ST_CHUNK];
    for (size_t f = 0; f < frames;) {
        size_t k = frames - f < ST_CHUNK ? frames - f : ST_CHUNK;
        for (size_t i = 0; i < k; i++) {
            const uint8_t *s = data + (f + i) * 2u * ch;  // left channel if stereo
            buf[i] = (int16_t)rd16(s);
        }
        feed(buf, k, &fed, pos, clip_start);
        f += k;
    }
    feed_noise(ST_TAIL_MS, &fed, pos, clip_start);
    pos->max_score = take_max_score();
    ww_reset();
}
