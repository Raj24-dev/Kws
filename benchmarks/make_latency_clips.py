"""SYNTHETIC latency stimuli: "Marvin" + a gap + a command, Windows voices. Because the parts are synthesized
separately, the end of the wake word is known to the sample (kw_end, after trimming the TTS silence).

    python make_latency_clips.py  -> sets/synthetic_latency.csv + audio/synthetic_latency/*.wav
"""
import csv
import itertools
import os

import numpy as np
import soundfile as sf

from make_tts import AUDIO, HERE, speak, voices

COMMANDS = ["turn on the lights", "what is the weather today", "play some music", "set an alarm for seven",
            "call my mother", "open the door", "what time is it", "stop the music", "increase the volume",
            "tell me a joke"]
RATES = [-2, 0, 2]
GAPS_MS = [150, 350]


def trimmed(path):
    x, _ = sf.read(path, dtype="int16")
    e = np.abs(x.astype(np.int32))
    idx = np.nonzero(e > 300)[0]  # ~-40 dBFS
    return x[max(0, idx[0] - 80): idx[-1] + 80]


def main():
    out_dir = os.path.join(AUDIO, "synthetic_latency")
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_tmp.wav")
    rows = []
    combos = list(itertools.product(sorted(voices().items()), RATES, GAPS_MS))
    for k, cmd in enumerate(COMMANDS * 2):
        (name, tok), rate, gap = combos[k % len(combos)]
        speak("Marvin", tmp, tok, rate)
        kw = trimmed(tmp)
        speak(cmd, tmp, tok, rate)
        c = trimmed(tmp)
        lead = np.zeros(8000, np.int16)  # 0.5 s
        x = np.concatenate([lead, kw, np.zeros(gap * 16, np.int16), c, np.zeros(8000, np.int16)])
        f = f"lat_{k:02d}_{name}_r{rate}_g{gap}.wav"
        sf.write(os.path.join(out_dir, f), x, 16000, subtype="PCM_16")
        rows.append({"file": f, "label": 1, "kw_end": round((len(lead) + len(kw)) / 16000, 4), "command": cmd,
                     "voice": name, "rate": rate, "gap_ms": gap})
    os.remove(tmp)
    with open(os.path.join(HERE, "sets", "synthetic_latency.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "clips")


if __name__ == "__main__":
    main()
