#!/usr/bin/env python3
"""Build an HTML report + spectrograms for recorded wake-word utterances.

Reads recordings/index.jsonl (written by ws_server.py), makes a small PNG
spectrogram per WAV, and writes recordings/report.html with one card per
utterance (audio player + spectrogram) plus a summary at the top.

    pip install numpy               (required; no matplotlib/PIL used)
    python report.py [recordings_dir]     # default: ../recordings next to this script
"""
import html
import json
import os
import statistics
import struct
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FFT_SIZE = 512
HOP = 160          # 10 ms at 16 kHz
PX_PER_FRAME = 2
DB_RANGE = 70.0     # clip range below the file's own max


def write_png(path, rgb):
    """rgb: (h, w, 3) uint8 array, row 0 = top. stdlib zlib + struct only."""
    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0 (none)
        raw += rgb[y].tobytes()
    compressed = zlib.compress(bytes(raw), 6)

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def spectrogram_png(wav_path, png_path, preroll_ms):
    import wave
    with wave.open(wav_path, "rb") as w:
        n = w.getnframes()
        audio = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64)
    if len(audio) < FFT_SIZE:
        audio = np.pad(audio, (0, FFT_SIZE - len(audio)))

    window = np.hanning(FFT_SIZE)
    n_frames = 1 + (len(audio) - FFT_SIZE) // HOP
    n_frames = max(n_frames, 1)
    mags = np.empty((n_frames, FFT_SIZE // 2 + 1))
    for i in range(n_frames):
        seg = audio[i * HOP: i * HOP + FFT_SIZE]
        if len(seg) < FFT_SIZE:
            seg = np.pad(seg, (0, FFT_SIZE - len(seg)))
        spec = np.fft.rfft(seg * window)
        mags[i] = np.abs(spec)

    mags = np.maximum(mags, 1e-9)
    db = 20 * np.log10(mags)
    db -= db.max()
    db = np.clip(db, -DB_RANGE, 0.0)
    level = (db + DB_RANGE) / DB_RANGE  # 0..1, 0 = quiet, 1 = loud

    # simple dark -> yellow colour map
    dark = np.array([15, 15, 30])
    bright = np.array([255, 225, 40])
    colors = dark + level[..., None] * (bright - dark)  # (frames, freq, 3)
    colors = colors.astype(np.uint8)

    # image: rows = freq bins low-to-high from bottom, cols = frames * PX_PER_FRAME
    n_freq = mags.shape[1]
    img = np.repeat(colors, PX_PER_FRAME, axis=0)          # (frames*2, freq, 3)
    img = np.flip(img, axis=1)                              # low freq -> bottom
    img = np.transpose(img, (1, 0, 2))                       # (freq, frames*2, 3)

    if preroll_ms:
        x = int(preroll_ms / 10 * PX_PER_FRAME)
        if 0 <= x < img.shape[1]:
            img[:, x] = [255, 255, 255]

    write_png(png_path, img)


def load_index(recordings_dir):
    rows = []
    seen_files = set()
    index_path = os.path.join(recordings_dir, "index.jsonl")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip bad lines
                rows.append(row)
                seen_files.add(row.get("file"))
    for name in sorted(os.listdir(recordings_dir)):
        if name.lower().endswith(".wav") and name not in seen_files:
            rows.append({"file": name})
    return rows


def fmt(v, suffix=""):
    return "-" if v is None else f"{v}{suffix}"


CARD_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }
.summary { background: #eee; border-radius: 8px; padding: 1em 1.5em; margin-bottom: 2em; }
.card { border: 1px solid #ccc; border-radius: 8px; padding: 1em; margin-bottom: 1em; }
.card h3 { margin: 0 0 0.3em; font-size: 1em; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 0.5em; }
audio { width: 100%; margin-bottom: 0.4em; }
img.spec { width: 100%; height: 130px; image-rendering: pixelated; display: block; }
.caption { color: #888; font-size: 0.8em; margin-top: 0.2em; }
.badge { font-weight: bold; }
.badge.ok { color: #0a7a0a; }
.badge.bad { color: #c22; }
@media (prefers-color-scheme: dark) {
  body { background: #111; color: #ddd; }
  .summary, .card { background: #1c1c1c; border-color: #333; }
  .meta, .caption { color: #999; }
  .badge.ok { color: #4ad14a; }
  .badge.bad { color: #f66; }
}
"""


def build_html(recordings_dir, rows):
    rows_sorted = sorted(rows, key=lambda r: r.get("time") or "", reverse=True)

    reasons = {}
    latencies, scores, durations = [], [], []
    n_verified = n_false = n_unknown = 0
    for r in rows:
        reasons[r.get("reason", "unknown")] = reasons.get(r.get("reason", "unknown"), 0) + 1
        if r.get("first_audio_latency_ms") is not None:
            latencies.append(r["first_audio_latency_ms"])
        if r.get("score") is not None:
            scores.append(r["score"])
        if r.get("duration_s") is not None:
            durations.append(r["duration_s"])
        v = r.get("verified")
        if v is True:
            n_verified += 1
        elif v is False:
            n_false += 1
        else:
            n_unknown += 1

    reason_line = ", ".join(f"{html.escape(str(k))}: {v}" for k, v in sorted(reasons.items()))
    summary = f"""
    <div class="summary">
      <b>{len(rows)}</b> utterances<br>
      by reason: {reason_line}<br>
      first_audio_latency_ms: median {statistics.median(latencies):.0f}, max {max(latencies):.0f}
      ({len(latencies)} known)<br>
      score: min {min(scores):.3f}, median {statistics.median(scores):.3f}, max {max(scores):.3f}
      ({len(scores)} known)<br>
      total audio: {sum(durations):.1f} s<br>
      verified: {n_verified} true, {n_false} false, {n_unknown} unknown
    </div>
    """ if rows else "<div class=\"summary\">No utterances found.</div>"

    cards = []
    for r in rows_sorted:
        wav = r.get("file", "?")
        png = f"spectrograms/{wav}.png"
        has_png = os.path.isfile(os.path.join(recordings_dir, "spectrograms", f"{wav}.png"))
        transcript = f'<div>transcript: "{html.escape(r["transcript"])}"</div>' if r.get("transcript") else ""

        verified = r.get("verified")
        if verified is True:
            badge = ' &middot; <span class="badge ok">&#10004; wake word verified</span>'
        elif verified is False:
            badge = ' &middot; <span class="badge bad">&#10008; false trigger (Whisper did not hear the wake word)</span>'
        else:
            badge = ""

        event_line = ""
        if r.get("event_peak") is not None and r.get("event_ms") is not None:
            event_line = f" &middot; score event peak {fmt(r.get('event_peak'))} over {fmt(r.get('event_ms'), ' ms')}"
        if r.get("resumes"):
            event_line += f" &middot; resumes {r['resumes']}"

        cards.append(f"""
    <div class="card">
      <h3>{html.escape(str(r.get('time', wav)))} &mdash; {html.escape(wav)}</h3>
      <div class="meta">
        score {fmt(r.get('score'))} &middot; duration {fmt(r.get('duration_s'), ' s')} &middot;
        reason {html.escape(str(r.get('reason', '-')))} &middot;
        latency {fmt(r.get('first_audio_latency_ms'), ' ms')} &middot;
        rms/peak {fmt(r.get('rms_dbfs'))}/{fmt(r.get('peak_dbfs'))} dBFS{badge}{event_line}
      </div>
      {transcript}
      <audio controls preload="none" src="{html.escape(wav)}"></audio>
      {f'<img class="spec" src="{html.escape(png)}">' if has_png else '<div class="caption">(no spectrogram)</div>'}
      <div class="caption">white line = detection moment (pre-roll before it)</div>
    </div>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Wake word recordings report</title>
<style>{CARD_CSS}</style></head>
<body>
<h1>Wake word recordings report</h1>
{summary}
{''.join(cards)}
</body></html>
"""


def main():
    recordings_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "recordings")
    recordings_dir = os.path.abspath(recordings_dir)
    spec_dir = os.path.join(recordings_dir, "spectrograms")
    os.makedirs(spec_dir, exist_ok=True)

    rows = load_index(recordings_dir)

    made = 0
    for r in rows:
        wav = r.get("file")
        if not wav:
            continue
        wav_path = os.path.join(recordings_dir, wav)
        png_path = os.path.join(spec_dir, f"{wav}.png")
        if not os.path.isfile(wav_path):
            continue
        if os.path.isfile(png_path) and os.path.getmtime(png_path) >= os.path.getmtime(wav_path):
            continue
        try:
            spectrogram_png(wav_path, png_path, r.get("preroll_ms"))
            made += 1
        except Exception as e:
            print(f"  warning: failed to make spectrogram for {wav}: {e}")

    report_path = os.path.join(recordings_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_html(recordings_dir, rows))

    print(f"Wrote {report_path}")
    print(f"Created {made} spectrogram PNG(s) ({len(rows)} utterances total)")


if __name__ == "__main__":
    main()
