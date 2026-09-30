// Self-test: runs the exact on-device pipeline (features + model + detection logic) on a recording that is
// compiled into the firmware (model/*.wav). This checks the model works on the chip, independent of the mic.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool ran;
    bool detected;
    float max_score;   // highest averaged probability
    int detect_ms;     // when it fired, measured from the start of the recording (-1 = never)
    int clip_ms;       // length of the recording
    char error[96];    // why the test could not run
} selftest_result_t;

// positive: the wake word recording (should detect). negative: 3 s of quiet noise (should NOT detect).
void wav_selftest_run(const uint8_t *wav, size_t len, selftest_result_t *positive, selftest_result_t *negative);

#ifdef __cplusplus
}
#endif
