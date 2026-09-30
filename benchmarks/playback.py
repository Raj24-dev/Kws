"""Plays 16 kHz clips through the PC speakers (WASAPI) and reports WHEN the first sample reached the DAC,
on the same clock as time.perf_counter() (PortAudio's stream time and perf_counter both use QueryPerformanceCounter;
the offset between them is measured on every call anyway)."""
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE_IN = 16000


def find_wasapi_speakers():
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and sd.query_hostapis(d["hostapi"])["name"] == "Windows WASAPI":
            return i, int(d["default_samplerate"])
    raise RuntimeError("no WASAPI output device")


DEVICE, RATE_OUT = find_wasapi_speakers()


def load16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    assert sr == RATE_IN, f"{path}: {sr} Hz (expected 16 kHz)"
    return x


def resample(x, factor_num, factor_den=1):
    """Band-limited (FFT) resampling by factor_num/factor_den."""
    n = len(x)
    m = n * factor_num // factor_den
    X = np.fft.rfft(x)
    Y = np.zeros(m // 2 + 1, np.complex128)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    return (np.fft.irfft(Y, m) * (m / n)).astype(np.float32)


def play(x16k, gain=1.0, pad_s=0.05):
    """Plays a 16 kHz float clip, blocking. Returns (t_dac0, t_end) in the perf_counter time base:
    t_dac0 = when sample 0 of the clip reaches the DAC (driver estimate)."""
    y = resample(np.concatenate([x16k, np.zeros(int(pad_s * RATE_IN), np.float32)]), RATE_OUT, RATE_IN) * gain
    y = np.clip(y, -1, 1)
    state = {"i": 0, "dac0": None}
    done = threading.Event()

    def cb(outdata, frames, tinfo, status):
        i = state["i"]
        if state["dac0"] is None:
            state["dac0"] = tinfo.outputBufferDacTime
        chunk = y[i:i + frames]
        outdata[:len(chunk), :] = chunk[:, None]
        outdata[len(chunk):, :] = 0
        state["i"] = i + frames
        if state["i"] >= len(y):
            raise sd.CallbackStop

    with sd.OutputStream(device=DEVICE, samplerate=RATE_OUT, channels=2, dtype="float32", callback=cb,
                         latency="low", finished_callback=done.set) as s:
        # stream clock -> perf_counter offset (median of a few reads)
        offs = []
        for _ in range(5):
            a = time.perf_counter()
            st = s.time
            b = time.perf_counter()
            offs.append(st - (a + b) / 2)
        off = float(np.median(offs))
        done.wait(len(y) / RATE_OUT + 5)
    return state["dac0"] - off, time.perf_counter()


class Looper:
    """Plays a clip in a loop in the background (to keep sound in the room for the CPU test)."""

    def __init__(self, x16k, gain=1.0):
        self.y = resample(x16k, RATE_OUT, RATE_IN) * gain
        self.i = 0

        def cb(outdata, frames, tinfo, status):
            n = len(self.y)
            idx = (np.arange(frames) + self.i) % n
            outdata[:, :] = self.y[idx][:, None]
            self.i = (self.i + frames) % n

        self.s = sd.OutputStream(device=DEVICE, samplerate=RATE_OUT, channels=2, dtype="float32", callback=cb)
        self.s.start()

    def stop(self):
        self.s.stop()
        self.s.close()
