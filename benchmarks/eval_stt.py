"""[HOST-ONLY] Server speech-to-text check on the REAL command recordings the server saved (preroll_ms 0 = exactly
what the board streamed): runs stt-services/main.py's transcribe_command and prints Whisper's per-segment
confidence (no_speech_prob, avg_logprob, compression_ratio), to tell hallucinated text from speech.

    python eval_stt.py            -> results/stt_segments.json
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REC = os.path.join(ROOT, "cloudServer", "server", "recordings")
sys.path.insert(0, os.path.join(ROOT, "cloudServer", "server", "stt-services"))


def main():
    import main as stt
    rows = [json.loads(l) for l in open(os.path.join(REC, "index.jsonl"), encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("preroll_ms") == 0 and r.get("stt_final_ms") is not None]
    out = []
    for r in rows:
        x, _ = sf.read(os.path.join(REC, r["file"]), dtype="float32")
        speech = stt.get_speech_timestamps(x, stt.VadOptions())
        segs, _ = stt.model.transcribe(x, language="en", beam_size=5, temperature=0.0, condition_on_previous_text=False,
                                       initial_prompt=stt.PUNCTUATION_PROMPT)
        segs = [{"text": s.text.strip(), "no_speech": round(s.no_speech_prob, 3), "logprob": round(s.avg_logprob, 3),
                 "cr": round(s.compression_ratio, 2), "start": s.start, "end": s.end} for s in segs]
        new = stt.transcribe_command((x * 32768).astype(np.int16).tobytes())
        out.append({"file": r["file"], "dur": r["duration_s"], "speech_s": round(sum(t["end"] - t["start"] for t in speech) / 16000, 2),
                    "saved_transcript": r["transcript"], "now": new, "segments": segs})
        print(out[-1]["file"][:22], out[-1]["dur"], out[-1]["speech_s"], "|", new[:70], "|",
              [(s["no_speech"], s["logprob"], s["cr"]) for s in segs], flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "stt_segments.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
