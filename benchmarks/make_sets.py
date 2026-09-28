"""Builds the frozen evaluation sets from REAL recordings made by the board's microphones (2026-09-27/28).

Source: cloudServer/server/recordings/*.wav whose index.jsonl entry has preroll_ms == 1000, i.e. the audio starts 1 s
before the board's detection and contains the word that triggered it, followed by what was said next.
Labels: index.jsonl "verified" (the server's former Whisper check) AND a fresh Whisper small.en pass here
(temperature 0, word timestamps). Clips where the two disagree are dropped as doubtful.
Anything used to train/validate the model (kws_s3/training_clips/*.zip) is excluded.

Keyword end (for the latency test) = end of the "Marvin" word: Whisper's word timing, refined to the last 10 ms
frame above (noise floor + 10 dB) within that word (+150 ms).

Split: stratified 50/50 validation/test, seed 26172. The test manifest is frozen with SHA-256 of every file.
Selection bias (disclosed): every clip exists because an earlier model fired on it, so these are "hard/known" cases,
and the positives are one speaker, one room, one device.

    python make_sets.py            -> sets/real_val.csv, sets/real_test.csv
"""
import csv
import glob
import hashlib
import json
import os
import random
import zipfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REC = os.path.join(ROOT, "cloudServer", "server", "recordings")
SEED = 26172


def edit1(a, b):
    """True if a and b differ by at most one edit"""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    return any(a == b[:i] + b[i + 1:] for i in range(len(b)))


def is_marvin(word):  # same rule as the server used for its labels
    w = "".join(c for c in word.lower() if c.isalpha())
    return w == "marvin" or (w[:4] == "marv" and edit1(w, "marvin"))


def kw_end_refine(x, w_start, w_end):
    fr = 160
    n = len(x) // fr
    e = 10 * np.log10(np.mean(x[:n * fr].reshape(n, fr).astype(np.float64) ** 2, axis=1) + 1e-12)
    thr = np.percentile(e, 20) + 10
    lo, hi = int(w_start * 100), min(n, int((w_end + 0.15) * 100))
    above = [i for i in range(lo, hi) if e[i] > thr]
    if not above:
        return w_end, "whisper"
    end = above[-1] + 1
    # the command may follow without a pause: then the energy end is not a word boundary -> keep Whisper's
    if end >= hi - 1:
        return w_end, "whisper"
    return end / 100, "energy"


def main():
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    trained = set()
    for z in glob.glob(os.path.join(ROOT, "kws_s3", "training_clips", "*.zip")):
        trained |= {os.path.basename(n) for n in zipfile.ZipFile(z).namelist()}
    rows = [json.loads(l) for l in open(os.path.join(REC, "index.jsonl"), encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        if r.get("preroll_ms") != 1000 or r.get("verified") is None or r["file"] in trained:
            continue
        path = os.path.join(REC, r["file"])
        if not os.path.exists(path):
            continue
        x, sr = sf.read(path, dtype="int16")
        segs, _ = model.transcribe(x.astype(np.float32) / 32768, language="en", beam_size=5, temperature=0.0,
                                   condition_on_previous_text=False, word_timestamps=True)
        words = [w for s in segs for w in s.words]
        # the word that triggered the board ends near 1.0 s (the detection point); look in the first 1.6 s
        hit = next((w for w in words if is_marvin(w.word) and w.start < 1.6), None)
        whisper_label = hit is not None
        row = {"file": r["file"], "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
               "duration_s": round(len(x) / sr, 2), "label_index": int(bool(r["verified"])),
               "label_whisper": int(whisper_label), "text_whisper": " ".join(w.word.strip() for w in words),
               "ref_transcript": (r.get("transcript") or "").replace("\n", " "), "orig_score": r.get("score"),
               "kw_start": "", "kw_end": "", "kw_end_method": ""}
        if hit:
            end, how = kw_end_refine(x, hit.start, hit.end)
            row.update(kw_start=round(hit.start, 3), kw_end=round(end, 3), kw_end_method=how)
        row["label"] = row["label_index"] if row["label_index"] == row["label_whisper"] else "doubtful"
        out.append(row)
        print(row["file"], row["label_index"], row["label_whisper"], row["kw_end"], row["kw_end_method"], row["text_whisper"][:60])

    keep = [r for r in out if r["label"] != "doubtful"]
    rnd = random.Random(SEED)
    for lab in (0, 1):
        grp = sorted((r for r in keep if r["label"] == lab), key=lambda r: r["file"])
        rnd.shuffle(grp)
        for i, r in enumerate(grp):
            r["split"] = "val" if i % 2 == 0 else "test"
    os.makedirs(os.path.join(HERE, "sets"), exist_ok=True)
    fields = ["file", "split", "label", "label_index", "label_whisper", "kw_start", "kw_end", "kw_end_method",
              "duration_s", "orig_score", "text_whisper", "ref_transcript", "sha256"]
    for split in ("val", "test"):
        with open(os.path.join(HERE, "sets", f"real_{split}.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(r for r in keep if r["split"] == split)
    with open(os.path.join(HERE, "sets", "real_doubtful.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(dict(r, split="excluded") for r in out if r["label"] == "doubtful")
    for split in ("val", "test"):
        s = [r for r in keep if r["split"] == split]
        print(split, "positives", sum(r["label"] == 1 for r in s), "negatives", sum(r["label"] == 0 for r in s))
    print("doubtful (excluded)", sum(r["label"] == "doubtful" for r in out))


if __name__ == "__main__":
    main()
