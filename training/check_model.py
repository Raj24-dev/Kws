#!/usr/bin/env python3
"""Runs a microWakeWord .tflite model on a WAV file with EXACTLY the same steps as the ESP32 firmware
(same features, same int8 conversion, same 3-frame streaming, same sliding-window average and cool-down,
and the same quiet-noise padding as the on-device self-test).

Use it to:
  * check a trained model before flashing it (does it fire on your recordings? does it stay quiet on others?)
  * compare with the SELF-TEST lines the ESP32 prints at boot. For the same model + WAV the numbers match.

Works in Google Colab right after training (everything is already installed there):
    !python check_model.py /content/drive/MyDrive/microwakeword/marvin/export/marvin.tflite my_recording.wav

On a laptop:  pip install numpy pymicro-features ai-edge-litert soundfile  (librosa optional, for resampling)
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # older setups
    from tensorflow.lite.python.interpreter import Interpreter
from pymicro_features import MicroFrontend

RATE = 16000
LEAD_MS, TAIL_MS, NEG_MS = 1200, 600, 3000     # same as main/wav_selftest.c
FEATURE_SCALE = 0.0390625                     # frontend integer output -> float units used in training
GATE_HOLD, LOOKBACK, GATE_RATIO = 160, 30, 2.0  # quiet-room gate, same as main/wake_word.cpp


class LcgNoise:
    """Deterministic quiet noise, identical to the firmware's self-test."""

    def __init__(self, seed):
        self.x = seed

    def take(self, n):
        out = np.empty(n, np.int16)
        x = self.x
        for i in range(n):
            x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
            out[i] = ((x >> 16) % 61) - 30
        self.x = x
        return out


def load_wav(path):
    try:
        import soundfile as sf
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        data = data[:, 0]
    except ImportError:
        import wave
        with wave.open(path) as w:
            rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            assert width == 2, "only 16-bit WAV without soundfile installed"
            data = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, ch)[:, 0] / 32768.0
    if rate != RATE:
        import librosa
        print(f"  (resampling {rate} Hz -> 16000 Hz)")
        data = librosa.resample(np.asarray(data, np.float32), orig_sr=rate, target_sr=RATE)
    return np.clip(np.round(np.asarray(data) * 32768.0), -32768, 32767).astype(np.int16)


class Detector:
    """Mirror of main/wake_word.cpp"""

    def __init__(self, model_path, cutoff, window, cooldown_ms=1000, gate=True):
        self.model_path = model_path
        self.gate = gate
        self._load()
        inp, out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        if len(inp["shape"]) != 3 or inp["shape"][2] != 40 or inp["dtype"] != np.int8:
            sys.exit(f"Not a streaming microWakeWord model: input {inp['shape']} {inp['dtype']}")
        self.inp, self.out = inp, out
        self.stride = int(inp["shape"][1])
        self.in_scale, self.in_zp = inp["quantization"]
        self.out_scale, self.out_zp = out["quantization"]
        self.cutoff, self.window, self.cooldown = cutoff, window, cooldown_ms // 10
        self.reset()

    def _load(self):
        self.it = Interpreter(model_path=self.model_path)   # fresh interpreter = all internal state cleared
        self.it.allocate_tensors()

    def reset(self):
        self._load()
        self.frontend = MicroFrontend()
        self.pending = np.zeros(0, np.int16)
        self.frames = []
        self.probs = [0.0] * self.window
        self.idx = 0
        self.ignore = -self.cooldown
        self.last_avg = 0.0
        self.max_avg = 0.0
        self.noise_rms, self.hold, self.open, self.paused, self.lookback = 3000.0, GATE_HOLD, True, 0, []
        self.inferences = self.skipped = 0

    def _feed(self, q, events):
        """One quantized frame into the model input; runs the model every `stride` frames."""
        self.frames.append(q)
        if len(self.frames) < self.stride:
            return
        self.it.set_tensor(self.inp["index"], np.stack(self.frames)[None, ...])
        self.frames = []
        self.it.invoke()
        self.inferences += 1
        p = (int(self.it.get_tensor(self.out["index"])[0][0]) - self.out_zp) * self.out_scale
        self.probs[self.idx] = p
        self.idx = (self.idx + 1) % self.window
        self.last_avg = sum(self.probs) / self.window
        self.max_avg = max(self.max_avg, self.last_avg)
        if self.ignore >= 0 and self.last_avg > self.cutoff:
            events.append(self.last_avg)
            self.probs = [0.0] * self.window
            self.idx = 0
            self.ignore = -self.cooldown

    def process(self, samples):
        """Feeds int16 samples (one 20 ms block, like the firmware); returns the detection scores."""
        events = []
        if self.gate:  # quiet-room gate: sound 6 dB above the background keeps the model running for 1.6 s
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
            n = self.noise_rms
            self.noise_rms = max(1.0, 0.9 * n + 0.1 * rms if rms < n else n * 1.002 + 0.01)
            if rms > self.noise_rms * GATE_RATIO:
                self.hold = GATE_HOLD
        self.pending = np.concatenate([self.pending, samples])
        while len(self.pending) >= 160:                     # the frontend works in 10 ms (160-sample) steps
            r = self.frontend.process_samples(self.pending[:160].tobytes())
            self.pending = self.pending[max(r.samples_read, 1):]
            if not r.features:
                continue
            f = np.asarray(r.features, np.float32)                   # already scaled by 0.0390625
            q = np.clip(np.rint(f / self.in_scale) + self.in_zp, -128, 127).astype(np.int8)
            self.lookback = (self.lookback + [q])[-LOOKBACK:]
            if not self.gate:
                self._feed(q, events)
            elif self.hold > 0:  # model running; after a pause first replay the paused frames (up to 300 ms)
                self.hold -= 1
                replay = 0 if self.open else min(self.paused, LOOKBACK - 1)
                for fr in self.lookback[-(replay + 1):]:
                    self._feed(fr, events)
                self.open, self.paused = True, 0
            else:  # quiet: model paused
                if self.open:
                    self.frames = []
                self.open = False
                self.paused += 1
                self.skipped += 1
            latest = self.probs[(self.idx - 1) % self.window]
            if self.ignore < 0 and latest < self.cutoff:
                self.ignore += 1
        return events


def run(det, chunks):
    """chunks: list of int16 arrays fed in 20 ms blocks. Returns (detections [(sample_index, score)], max_avg)."""
    det.reset()
    fed, found = 0, []
    for c in chunks:
        for off in range(0, len(c), 320):
            blk = c[off:off + 320]
            for score in det.process(blk):
                found.append((fed + len(blk), score))
            fed += len(blk)
    return found, det.max_avg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help=".tflite file (Colab Step 13 export)")
    ap.add_argument("wav", nargs="*", help="recordings to test (any sample rate; mono or stereo)")
    ap.add_argument("--cutoff", type=float, help="threshold (default: from the .json next to the model, else 0.9)")
    ap.add_argument("--window", type=int, help="sliding window (default: from the .json, else 5)")
    ap.add_argument("--no-gate", action="store_true", help="run the model on every frame (firmware before the gate)")
    a = ap.parse_args()

    cutoff, window = 0.9, 5
    js = os.path.splitext(a.model)[0] + ".json"
    if os.path.exists(js):
        micro = json.load(open(js)).get("micro", {})
        cutoff, window = micro.get("probability_cutoff", cutoff), micro.get("sliding_window_size", window)
    cutoff = a.cutoff if a.cutoff is not None else cutoff
    window = a.window if a.window is not None else window
    det = Detector(a.model, cutoff, window, gate=not a.no_gate)
    print(f"model {os.path.basename(a.model)}: stride {det.stride}, input scale {det.in_scale:.5f} zp {det.in_zp}, "
          f"threshold {cutoff:.2f}, window {window}")

    found, mx = run(det, [LcgNoise(2).take(NEG_MS * 16)])
    print(f"noise only (3 s)   : {'FAIL - false detection!' if found else 'PASS - stayed quiet'}  (highest score {mx:.2f})")

    for path in a.wav:
        clip = load_wav(path)
        noise = LcgNoise(1)
        lead = noise.take(LEAD_MS * 16)
        found, mx = run(det, [lead, clip, noise.take(TAIL_MS * 16)])
        clip_ms = len(clip) * 1000 // RATE
        if found:
            t = (found[0][0] - len(lead)) * 1000 // RATE
            print(f"{os.path.basename(path)}: PASS - detected (highest score {mx:.2f}, fired {t} ms after the "
                  f"recording started; recording is {clip_ms} ms)")
        else:
            print(f"{os.path.basename(path)}: FAIL - not detected (highest score {mx:.2f}, threshold {cutoff:.2f})")


if __name__ == "__main__":
    main()
