#!/usr/bin/env python3
"""Cut trigger-word clips out of recorded wake-word utterances for training.

Reads one or more recordings/index.jsonl (written by the cloud server),
keeps records with preroll_ms >= 800 (long enough pre-roll to contain the whole wake
word), labels each true/false (already-verified records, or by asking a speech-to-text
service whether the wake word appears in the first few words), cuts the pre-roll +
tail_ms out of the WAV, and zips the results for the training notebook.

    python export_training_clips.py <recordings_dir> [<recordings_dir> ...] [--out DIR]
                                     [--stt http://127.0.0.1:8000/transcribe] [--tail-ms 250]
"""
import argparse
import json
import mimetypes
import os
import re
import shutil
import sys
import urllib.request
import uuid
import wave
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_PREROLL_MS = 800


def heard_wake_word(text, wake_word):
    """Same rule the cloud server uses: wake word among the first 4 words; a word that keeps its first 4 letters
    and differs by one letter counts too (Whisper writes a real "Marvin" as "Marvyn" at times)."""
    ww = wake_word.lower()
    words = re.sub(r"[^a-z']+", " ", text.lower()).split()
    words = [w[:-2] if w.endswith("'s") else w for w in words[:4]]
    return any(w == ww or (w.startswith(ww[:4]) and one_edit_apart(w, ww)) for w in words)


def one_edit_apart(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i, j = i + 1, j + 1
            continue
        edits += 1
        if edits > 1:
            return False
        i += len(a) >= len(b)
        j += len(b) >= len(a)
    return edits + (len(a) - i) + (len(b) - j) <= 1


def transcribe(stt_url, wav_path):
    """POST the WAV to the STT service as multipart/form-data. stdlib only, no requests."""
    boundary = uuid.uuid4().hex
    with open(wav_path, "rb") as f:
        data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(wav_path)}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        stt_url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["text"]


def load_records(recordings_dir):
    index_path = os.path.join(recordings_dir, "index.jsonl")
    if not os.path.isfile(index_path):
        return []
    rows = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def cut_clip(src_path, dst_path, ms):
    with wave.open(src_path, "rb") as w:
        rate = w.getframerate()
        n = min(w.getnframes(), int(rate * ms / 1000))
        frames = w.readframes(n)
        with wave.open(dst_path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(frames)


def zip_dir(zip_path, src_dir):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(src_dir)):
            z.write(os.path.join(src_dir, name), name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings_dirs", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "training_clips"))
    ap.add_argument("--stt", default=None, help="speech-to-text URL, e.g. http://127.0.0.1:8000/transcribe")
    ap.add_argument("--tail-ms", type=int, default=250)
    ap.add_argument("--relabel", action="store_true",
                    help="ignore the stored verified labels and transcribe every clip again with --stt")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    pos_dir = os.path.join(out_dir, "positives")
    neg_dir = os.path.join(out_dir, "hard_negatives")
    for d in (pos_dir, neg_dir):  # start clean: a relabelled clip must not stay in the other folder
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    n_pos = n_neg = n_skip = 0
    for recordings_dir in args.recordings_dirs:
        recordings_dir = os.path.abspath(recordings_dir)
        for r in load_records(recordings_dir):
            preroll_ms = r.get("preroll_ms")
            wav = r.get("file")
            if not wav or preroll_ms is None or preroll_ms < MIN_PREROLL_MS:
                continue
            wav_path = os.path.join(recordings_dir, wav)
            if not os.path.isfile(wav_path):
                continue

            label = None if args.relabel else r.get("verified")
            transcript = r.get("transcript")
            if label is None and args.stt:
                try:
                    transcript = transcribe(args.stt, wav_path)
                    label = heard_wake_word(transcript, r.get("wake_word") or "marvin")
                except Exception as e:
                    print(f"  warning: STT failed for {wav}: {e}")

            if label is None:
                n_skip += 1
                print(f"? {wav}  (no label, skipped)")
                continue

            dst_dir = pos_dir if label else neg_dir
            cut_clip(wav_path, os.path.join(dst_dir, wav), preroll_ms + args.tail_ms)
            n_pos += bool(label)
            n_neg += not label
            mark = "+" if label else "-"
            extra = f'  "{transcript}"' if transcript else ""
            print(f"{mark} {wav}{extra}")

    pos_zip = os.path.join(out_dir, "device_positives.zip")
    neg_zip = os.path.join(out_dir, "device_hard_negatives.zip")
    zip_dir(pos_zip, pos_dir)
    zip_dir(neg_zip, neg_dir)

    print(f"\npositives: {n_pos}, hard negatives: {n_neg}, skipped (no label): {n_skip}")
    print(f"wrote {pos_zip}")
    print(f"wrote {neg_zip}")
    print("Listen to a few before training: Whisper can mishear a distant 'Marvin'. "
          "Upload the zips to Drive and set MY_RECORDINGS_ZIP / MY_NEGATIVES_ZIP in the notebook.")


if __name__ == "__main__":
    sys.exit(main())
