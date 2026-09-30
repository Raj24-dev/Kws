"""[HOST-ONLY] accuracy with training/check_model.py, the PC mirror of the firmware pipeline (same features,
int8 conversion, 3-frame streaming, sliding-window average, cool-down and quiet-room gate).

    python run_host.py sweep  --label baseline          threshold sweep on the VALIDATION sets -> picks a cutoff
    python run_host.py test   --label baseline --cutoff 0.5   frozen TEST sets at one cutoff (run once per model)
    python run_host.py faph   --label baseline --cutoff 0.5 <long negative wav ...>   false accepts per hour

Every clip is fed like the firmware's self-test: 1.2 s quiet noise, the clip, 0.6 s quiet noise (detector reset per clip).
Per clip we keep the highest sliding-window score (detection at cutoff c <=> max score > c; measured with cutoff 1.01 so
no reset hides a later peak).
Threshold rule (fixed before looking at test data): the highest cutoff whose validation TPR is still >= 95 % on the
real positives; ties impossible. Rationale: the PS asks for high TPR AND near-zero false activations; with TPR pinned,
a higher cutoff can only lower false accepts.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
from check_model import LEAD_MS, TAIL_MS, Detector, LcgNoise, load_wav, run  # noqa: E402

REC = os.path.join(ROOT, "server", "recordings")
OLD_TTS = os.path.join(HERE, "audio", "synthetic_tts_2026-09-27")  # previous session's TTS set (validation only)
MODEL = os.path.join(ROOT, "firmware", "model", "marvin.tflite")
CUTS = [round(c, 2) for c in np.arange(0.30, 0.99, 0.05)]


def real_set(split):
    rows = list(csv.DictReader(open(os.path.join(HERE, "sets", f"real_{split}.csv"), encoding="utf-8")))
    return [(os.path.join(REC, r["file"]), int(r["label"])) for r in rows]


def folder_set(path, pos_key="marvin"):
    return [(p, int(pos_key in os.path.basename(p).lower() and not os.path.basename(p).startswith("neg")))
            for p in sorted(glob.glob(os.path.join(path, "*.wav")))]


def sets_for(kind):
    if kind == "val":
        s = {"real_val": real_set("val")}
        if os.path.isdir(OLD_TTS):
            s["synthetic_tts_2026-09-27"] = folder_set(OLD_TTS)
        return s
    return {"real_test": real_set("test"),
            "synthetic_confusable": folder_set(os.path.join(HERE, "audio", "synthetic_confusable")),
            "synthetic_marvin": folder_set(os.path.join(HERE, "audio", "synthetic_marvin"))}


def max_scores(det, items):
    out = []
    for path, label in items:
        clip = load_wav(path)
        noise = LcgNoise(1)
        _, mx = run(det, [noise.take(LEAD_MS * 16), clip, noise.take(TAIL_MS * 16)])
        out.append((os.path.basename(path), label, round(float(mx), 4)))
    return out


def rates(scores, c):
    pos = [s for _, l, s in scores if l == 1]
    neg = [s for _, l, s in scores if l == 0]
    return {"cutoff": c, "tpr": round(sum(s > c for s in pos) / len(pos), 4) if pos else None,
            "far": round(sum(s > c for s in neg) / len(neg), 4) if neg else None, "n_pos": len(pos), "n_neg": len(neg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "test", "faph"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    det = Detector(a.model, 1.01, a.window)  # never fires: max score per clip, threshold-free

    if a.mode in ("sweep", "test"):
        res = {"label": a.label, "evidence": "HOST-ONLY", "model": os.path.basename(a.model), "window": a.window, "sets": {}}
        for name, items in sets_for("val" if a.mode == "sweep" else "test").items():
            sc = max_scores(det, items)
            res["sets"][name] = {"scores": sc, "sweep": [rates(sc, c) for c in CUTS],
                                 "at_cutoff": rates(sc, a.cutoff)}
            r = res["sets"][name]["at_cutoff"]
            print(f"{name:28s} n+={r['n_pos']:4d} n-={r['n_neg']:4d}  at {a.cutoff:.2f}: TPR {r['tpr']}  FAR {r['far']}")
        if a.mode == "sweep":
            sw = res["sets"]["real_val"]["sweep"]
            ok = [r for r in sw if r["tpr"] >= 0.95]
            res["chosen_cutoff"] = max(r["cutoff"] for r in ok) if ok else min(CUTS)
            print("validation sweep (real_val):")
            for r in sw:
                print(f"  {r['cutoff']:.2f}  TPR {r['tpr']:.3f}  FAR {r['far']:.3f}")
            print("chosen cutoff (rule: highest with real_val TPR >= 0.95):", res["chosen_cutoff"])
        out = os.path.join(HERE, "results", f"{a.label}_host_{a.mode}.json")
        json.dump(res, open(out, "w"), indent=1)
        print("->", out)
    else:
        det = Detector(a.model, a.cutoff, a.window)
        total_s, fas, per = 0.0, 0, []
        for p in a.files:
            x = load_wav(p)
            found, mx = run(det, [x])
            total_s += len(x) / 16000
            fas += len(found)
            per.append({"file": os.path.basename(p), "hours": round(len(x) / 16000 / 3600, 3), "false_accepts": len(found),
                        "at_s": [round(i / 16000, 1) for i, _ in found]})
            print(per[-1])
        res = {"label": a.label, "evidence": "HOST-ONLY", "cutoff": a.cutoff, "hours": round(total_s / 3600, 3),
               "false_accepts": fas, "faph": round(fas / (total_s / 3600), 2), "files": per}
        json.dump(res, open(os.path.join(HERE, "results", f"{a.label}_host_faph_{a.cutoff}.json"), "w"), indent=1)
        print(json.dumps({k: res[k] for k in ("hours", "false_accepts", "faph")}))


if __name__ == "__main__":
    main()
