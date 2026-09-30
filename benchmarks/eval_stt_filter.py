"""[HOST-ONLY] Does the STT confidence filter (stt-services/main.py is_speech) drop real speech?
Runs every saved recording with a 1 s pre-roll from the detection point on (= the command, as the server transcribes
it), keeps the raw per-segment output, and reports the transcripts with and without the filter.

    python eval_stt_filter.py   -> results/stt_filter_check.json
"""
import json
import os
import sys

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REC = os.path.join(ROOT, "server", "recordings")
sys.path.insert(0, os.path.join(ROOT, "server", "stt-services"))


def main():
    import main as stt
    rows = [json.loads(l) for l in open(os.path.join(REC, "index.jsonl"), encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("preroll_ms") == 1000 and os.path.exists(os.path.join(REC, r["file"]))]
    out, changed = [], []
    for r in rows:
        x, _ = sf.read(os.path.join(REC, r["file"]), dtype="float32")
        x = x[16000:]
        if not len(x) or not stt.get_speech_timestamps(x, stt.VadOptions()):
            continue
        segs = list(stt.model.transcribe(x, language="en", beam_size=5, temperature=0.0,
                                         condition_on_previous_text=False, initial_prompt=stt.PUNCTUATION_PROMPT)[0])
        old = stt.tidy(" ".join(s.text.strip() for s in segs).strip())
        new = stt.tidy(" ".join(s.text.strip() for s in segs if stt.is_speech(s)).strip())
        o = {"file": r["file"], "old": old, "new": new,
             "segments": [(s.text.strip(), round(s.compression_ratio, 2), round(s.avg_logprob, 3)) for s in segs]}
        out.append(o)
        if old != new:
            changed.append(o)
            print("CHANGED", o["file"][:22], "|", old[:80], "->", new[:60], "|", o["segments"][:3], flush=True)
    res = {"evidence": "HOST-ONLY", "utterances_with_speech": len(out), "changed_by_filter": len(changed),
           "changed": changed, "all": out}
    json.dump(res, open(os.path.join(HERE, "results", "stt_filter_check.json"), "w"), indent=1)
    print(len(out), "utterances with speech;", len(changed), "changed by the filter")


if __name__ == "__main__":
    main()
