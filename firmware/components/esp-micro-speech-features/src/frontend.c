/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "frontend.h"

#include "bits.h"
#include "sdkconfig.h"

#if CONFIG_KWS_PROFILE_OPS  // benchmark builds: CPU cycles per stage (read by the firmware's [prof] line)
#include "esp_cpu.h"
uint32_t g_frontend_prof_cycles[8];
#define PROF_MARK(k)                                       do {                                                       uint32_t now_ = esp_cpu_get_cycle_count();               g_frontend_prof_cycles[k] += now_ - prof_t_;             prof_t_ = now_;                                        } while (0)
#else
#define PROF_MARK(k)   do {                 } while (0)
#endif

struct FrontendOutput FrontendProcessSamples(struct FrontendState *state, const int16_t *samples, size_t num_samples,
                                             size_t *num_samples_read) {
  struct FrontendOutput output;
  output.values = NULL;
  output.size = 0;

#if CONFIG_KWS_PROFILE_OPS
  uint32_t prof_t_ = esp_cpu_get_cycle_count();
#endif
  // Try to apply the window - if it fails, return and wait for more data.
  if (!WindowProcessSamples(&state->window, samples, num_samples, num_samples_read)) {
    return output;
  }
  PROF_MARK(0);

  // Apply the FFT to the window's output (and scale it so that the fixed point
  // FFT can have as much resolution as possible).
  int input_shift = 15 - MostSignificantBit32(state->window.max_abs_output_value);
  FftCompute(&state->fft, state->window.output, input_shift);
  PROF_MARK(1);

  // We can re-ruse the fft's output buffer to hold the energy.
  int32_t *energy = (int32_t *) state->fft.output;

  FilterbankConvertFftComplexToEnergy(&state->filterbank, state->fft.output, energy);
  PROF_MARK(2);

  FilterbankAccumulateChannels(&state->filterbank, energy);
  PROF_MARK(3);
  uint32_t *scaled_filterbank = FilterbankSqrt(&state->filterbank, input_shift);
  PROF_MARK(4);

  // Apply noise reduction.
  NoiseReductionApply(&state->noise_reduction, scaled_filterbank);
  PROF_MARK(5);

  if (state->pcan_gain_control.enable_pcan) {
    PcanGainControlApply(&state->pcan_gain_control, scaled_filterbank);
  }
  PROF_MARK(6);

  // Apply the log and scale.
  int correction_bits = MostSignificantBit32(state->fft.fft_size) - 1 - (kFilterbankBits / 2);
  uint16_t *logged_filterbank =
      LogScaleApply(&state->log_scale, scaled_filterbank, state->filterbank.num_channels, correction_bits);
  PROF_MARK(7);

  output.size = state->filterbank.num_channels;
  output.values = logged_filterbank;
  return output;
}

void FrontendReset(struct FrontendState *state) {
  WindowReset(&state->window);
  FftReset(&state->fft);
  FilterbankReset(&state->filterbank);
  NoiseReductionReset(&state->noise_reduction);
}

