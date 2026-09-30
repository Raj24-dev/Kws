"""[HOST-ONLY] Validation-only sweep of sliding window x cutoff: TPR / hard-negative FAR on real_val, false accepts
per hour on the VALIDATION half of the synthetic long speech (David voice; the Zira half is kept for testing), and
detection delay after the annotated keyword end (latency driver).

    python sweep_window.py -> results/sweep_window_val.json
"""
import csv
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
from check_model import LEAD_MS, TAIL_MS, Detector, LcgNoise, load_wav, run  # noqa: E402

MODEL = os.path.join(ROOT, "firmware", "model", "marvin.tflite")
REC = os.path.join(ROOT, "server", "recordings")
VAL_SPEECH = os.path.join(HERE, "audio", "synthetic_longspeech_00_David_r-3.wav")


def main():
    val = list(csv.DictReader(open(os.path.join(HERE, "sets", "real_val.csv"), encoding="utf-8")))
    clips = [(load_wav(os.path.join(REC, r["file"])), int(r["label"]), float(r["kw_end"]) if r["kw_end"] else None) for r in val]
    speech = load_wav(VAL_SPEECH)
    hours = len(speech) / 16000 / 3600
    out = []
    for window in (3, 4, 5):
        for cutoff in (0.5, 0.6, 0.7, 0.8, 0.9):
            det = Detector(MODEL, cutoff, window)
            tp = fa = 0
            delays = []
            for x, lab, ke in clips:
                noise = LcgNoise(1)
                found, _ = run(det, [noise.take(LEAD_MS * 16), x, noise.take(TAIL_MS * 16)])
                if lab == 1 and found:
                    tp += 1
                    if ke is not None:
                        delays.append((found[0][0] - LEAD_MS * 16) / 16 - ke * 1000)
                fa += lab == 0 and bool(found)
            found, _ = run(det, [speech])
            npos = sum(l == 1 for _, l, _ in clips)
            nneg = len(clips) - npos
            r = {"window": window, "cutoff": cutoff, "tpr": round(tp / npos, 3), "hard_far": round(fa / nneg, 3),
                 "faph_val_speech": round(len(found) / hours, 2), "delay_ms_median": round(st.median(delays)) if delays else None}
            out.append(r)
            print(r, flush=True)
    json.dump({"evidence": "HOST-ONLY", "val_speech_hours": round(hours, 2), "rows": out},
              open(os.path.join(HERE, "results", "sweep_window_val.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
