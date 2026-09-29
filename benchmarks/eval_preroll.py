"""[HOST-ONLY] Pre-roll design check with the server's real transcription code (stt-services/main.py).

For every real positive with a command: the firmware mirror gives the detection point d (cutoff from the model json).
  A  today : stream starts at d, no pre-roll          -> transcribe_command(x[d:])
  B  fix   : stream starts at d - P, pre-roll P       -> transcribe_command(x[d-P:], detect_s=P)
  R  ideal : audio from the annotated keyword end     -> transcribe_command(x[kw_end:])
Word error rate of A and B against R, and how often the wake word leaks into B's transcript.

    python eval_preroll.py val 0.25     (design choice on validation; run 'test' once for the report)
"""
import csv
import json
import os
import re
import sys

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "cloudServer", "server", "stt-services"))
from check_model import Detector, LcgNoise, run  # noqa: E402
from summarize_acoustic import wer  # noqa: E402

REC = os.path.join(ROOT, "cloudServer", "server", "recordings")
MODEL = os.path.join(ROOT, "kws_s3", "model", "marvin.tflite")
LEAD = 1200 * 16


def main(split, preroll_s):
    import main as stt
    cutoff = json.load(open(MODEL.replace(".tflite", ".json")))["micro"]["probability_cutoff"]
    det = Detector(MODEL, cutoff, 5)
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "sets", f"real_{split}.csv"), encoding="utf-8"))
            if r["label"] == "1" and r["kw_end"]]
    out, ea, eb, n, leaks = [], 0, 0, 0, 0
    for r in rows:
        x, _ = sf.read(os.path.join(REC, r["file"]), dtype="int16")
        found, _ = run(det, [LcgNoise(1).take(LEAD), x])
        if not found:
            continue
        d = (found[0][0] - LEAD) / 16000
        ref = stt.transcribe_command(x[int(float(r["kw_end"]) * 16000):].tobytes())
        if not ref:
            continue  # nothing said after the wake word
        a = stt.transcribe_command(x[int(d * 16000):].tobytes())
        p = min(preroll_s, d)
        b = stt.transcribe_command(x[int((d - p) * 16000):].tobytes(), detect_s=p)
        e1, m = wer(ref, a)
        e2, _ = wer(ref, b)
        leak = bool(re.match(r"\W*(marv|arvin|vin\b)", b.lower()))
        ea, eb, n, leaks = ea + e1, eb + e2, n + m, leaks + leak
        out.append({"file": r["file"], "detect_after_kw_ms": round((d - float(r["kw_end"])) * 1000), "ref": ref,
                    "no_preroll": a, "preroll": b, "wer_no_preroll": round(e1 / m, 3), "wer_preroll": round(e2 / m, 3),
                    "wake_word_leak": leak})
        print(out[-1], flush=True)
    res = {"evidence": "HOST-ONLY", "split": split, "cutoff": cutoff, "preroll_s": preroll_s, "commands": len(out),
           "wer_no_preroll": round(ea / n, 4) if n else None, "wer_preroll": round(eb / n, 4) if n else None,
           "wake_word_leaks": leaks, "items": out}
    json.dump(res, open(os.path.join(HERE, "results", f"preroll_eval_{split}_{int(preroll_s * 1000)}ms.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
