"""[HOST-ONLY] What the stream loses when it starts at the detection instead of at the end of the wake word.

For every real positive of a set: the firmware mirror (check_model.py) gives the detection point d (end of the 20 ms
block in which it fired, like the board); kw_end comes from the set's annotation. Then the command is transcribed by
the server's own code (stt-services/main.py: small.en, same prompt + VAD gate) twice:
   A) audio from d          (firmware today: CONFIG_KWS_PREROLL_MS=0)
   B) audio from kw_end     (nothing lost)
and the words present in B but missing at the start of A are counted.

    python analyze_preroll.py val     -> results/preroll_val.json  (design decisions are made on val only)
"""
import csv
import json
import os
import re
import statistics as st
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "cloudServer", "server", "stt-services"))
from check_model import Detector, LcgNoise, run  # noqa: E402

REC = os.path.join(ROOT, "cloudServer", "server", "recordings")
LEAD = 1200 * 16


def words(t):
    return re.findall(r"[a-z']+", t.lower())


def main(split, cutoff=0.5):
    import main as stt  # loads Whisper small.en (the server's code, unchanged)
    det = Detector(os.path.join(ROOT, "kws_s3", "model", "marvin.tflite"), cutoff, 5)
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "sets", f"real_{split}.csv"), encoding="utf-8"))
            if r["label"] == "1" and r["kw_end"]]
    out = []
    for r in rows:
        x, _ = sf.read(os.path.join(REC, r["file"]), dtype="int16")
        found, _ = run(det, [LcgNoise(1).take(LEAD), x])
        if not found:
            continue
        d = (found[0][0] - LEAD) / 16000
        ke = float(r["kw_end"])
        a = stt.transcribe_command(x[int(d * 16000):].tobytes())
        b = stt.transcribe_command(x[int(ke * 16000):].tobytes())
        wa, wb = words(a), words(b)
        missing = [w for w in wb if w not in wa]  # words heard only when the stream starts at the keyword end
        out.append({"file": r["file"], "kw_end": ke, "detect": round(d, 3), "delay_ms": round((d - ke) * 1000),
                    "from_detection": a, "from_kw_end": b, "missing_words": missing,
                    "first_word_lost": bool(wb) and wb[0] not in wa})
        print(out[-1], flush=True)
    delays = [o["delay_ms"] for o in out]
    with_cmd = [o for o in out if words(o["from_kw_end"])]
    res = {"evidence": "HOST-ONLY", "split": split, "cutoff": cutoff, "n": len(out),
           "delay_ms": {"median": st.median(delays), "p90": sorted(delays)[int(0.9 * (len(delays) - 1))],
                        "min": min(delays), "max": max(delays)},
           "utterances_with_command": len(with_cmd),
           "commands_first_word_lost_without_preroll": sum(o["first_word_lost"] for o in with_cmd),
           "commands_any_word_missing_without_preroll": sum(bool(o["missing_words"]) for o in with_cmd),
           "items": out}
    json.dump(res, open(os.path.join(HERE, "results", f"preroll_{split}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
