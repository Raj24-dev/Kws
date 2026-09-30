"""Acoustic end-to-end test: plays recordings through the PC speakers to the board's microphones.  [DEVICE-ACOUSTIC]

    python run_acoustic.py --label baseline sets/real_test.csv [more manifests or folders of .wav ...]

Needs: the board on COM6 (reset at start), server.js (port 3000) + stt-services on THIS PC, speakers near the board.
For every clip it records
  * detections (board serial log), the server's "start" / "first_audio" / "final" (transcript) events, all stamped
    with time.perf_counter() on this PC;
  * t_dac0 = when the clip's first sample reached the speaker DAC (WASAPI estimate, same clock).
Latency (PS metric) = first audio frame received by the ASR server - end of the keyword in the played clip
  = t_first_audio - (t_dac0 + kw_end + 1.5 ms acoustic path at ~0.5 m). One clock (this PC); the error bar is the
  driver's DAC-time accuracy (a few ms) + the /live event hop (localhost, < 1 ms).
Decomposition: device "first audio sent N ms after the detection" (board clock), network one-way = ping RTT / 2,
  detection delay = the rest.
Positives: detected or missed (TPR). Negatives: any detection is a false accept.
Writes results/<label>_acoustic_<set>.json.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as stt
import subprocess
import threading
import time

import numpy as np
from websockets.sync.client import connect

from devlog import DeviceLog, summarize_status
from playback import load16k, play

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(os.path.dirname(HERE), "server", "recordings")
ACOUSTIC_S = 0.0015
TARGET_DBFS = -20.0  # speech level of every clip before --gain (95th percentile of 10 ms frame RMS)


def load_items(args):
    items = []
    for src in args:
        if src.endswith(".csv"):
            audio_dir = os.path.join(HERE, "audio", os.path.splitext(os.path.basename(src))[0])
            for r in csv.DictReader(open(src, encoding="utf-8")):
                path = os.path.join(audio_dir if os.path.isdir(audio_dir) else REC, r["file"])
                items.append({"path": path, "label": int(r["label"]),
                              "kw_end": float(r["kw_end"]) if r["kw_end"] else None, "set": os.path.basename(src),
                              "ref": r.get("ref_transcript", "")})
        else:  # folder of wavs: label from the name (marvin -> 1)
            for p in sorted(glob.glob(os.path.join(src, "*.wav"))):
                items.append({"path": p, "label": int("marvin" in os.path.basename(p).lower()), "kw_end": None,
                              "set": os.path.basename(os.path.normpath(src)), "ref": ""})
    return items


def normalize(x, level_db):
    """Speech level (95th percentile of 10 ms frame RMS) -> level_db dBFS, peaks limited to -1 dBFS."""
    fr = x[:len(x) // 160 * 160].reshape(-1, 160)
    lvl = np.percentile(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9, 95)
    g = min(10 ** (level_db / 20) / lvl, 0.89 / (np.max(np.abs(x)) + 1e-9))
    return x * g


class Live:
    """Collects the server's /live events with local arrival times."""

    def __init__(self, url="ws://127.0.0.1:3000/live"):
        self.events = []
        self.ws = connect(url, max_size=None)
        self.ws.recv()  # hello + history
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            for msg in self.ws:
                t = time.perf_counter()
                m = json.loads(msg)
                if m.get("type") in ("start", "first_audio", "final"):
                    m["t"] = t
                    self.events.append(m)
        except Exception:
            pass


def ping_rtt_ms(ip, n=20):
    out = subprocess.run(["ping", "-n", str(n), "-l", "32", ip], capture_output=True, text=True).stdout
    v = [float(x) for x in re.findall(r"time[=<](\d+)ms", out)]
    return {"median": stt.median(v), "min": min(v), "max": max(v), "n": len(v)} if v else None


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--level-db", type=float, default=TARGET_DBFS, help="digital speech level of the clips (dBFS)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="", help="result name (default: first source)")
    ap.add_argument("--offline", action="store_true", help="no Wi-Fi/server: detections only (no latency, no transcripts)")
    a = ap.parse_args()

    items = load_items(a.sources)
    if a.limit:
        items = items[:a.limit]
    name = a.name or os.path.splitext(os.path.basename(os.path.normpath(a.sources[0])))[0]
    base = os.path.join(HERE, "results", f"{a.label}_acoustic_{name}")
    os.makedirs(os.path.dirname(base), exist_ok=True)

    dev = DeviceLog(a.port, base + ".log", reset=True)
    ip, rtt = None, None
    if a.offline:
        dev.wait_for(r"listening for", 30)
        live = type("NoLive", (), {"events": []})()
    else:
        ip_line = dev.wait_for(r"connected, IP address", 60)
        ip = re.search(r"(\d+\.\d+\.\d+\.\d+)", ip_line).group(1) if ip_line else None
        if not dev.wait_for(r"connected to the server", 60):
            print("WARNING: board did not report a server connection")
        live = Live()
        rtt = ping_rtt_ms(ip) if ip else None
    time.sleep(3)  # detector warm-up (first second is ignored by design)
    print(f"board {ip}, ping RTT {rtt}", flush=True)

    trials = []
    for k, it in enumerate(items):
        x = normalize(load16k(it["path"]), a.level_db)
        # wait until the board is idle: no stream in progress and >= 1.5 s since its last detection
        t_wait = time.perf_counter()
        while time.perf_counter() - t_wait < 25:
            ev = list(live.events)
            open_streams = len([e for e in ev if e["type"] == "start"]) - len([e for e in ev if e["type"] == "final"])
            last_det = max([d["t"] for d in dev.snapshot()["detections"]], default=0)
            if open_streams <= 0 and time.perf_counter() - last_det > 1.5:
                break
            time.sleep(0.2)
        n_ev, n_det, n_sev = len(live.events), len(dev.snapshot()["detections"]), len(dev.snapshot()["events"])
        t_dac0, t_end = play(x, a.gain)
        time.sleep(2.0)  # late detections / stream start
        det = dev.snapshot()["detections"][n_det:]
        det = [d for d in det if d["t"] >= t_dac0]
        # a stream was started: wait for its transcript
        t_w = time.perf_counter()
        while time.perf_counter() - t_w < 20:
            ev = live.events[n_ev:]
            if not any(e["type"] == "start" for e in ev) or any(e["type"] == "final" for e in ev):
                break
            time.sleep(0.2)
        ev = live.events[n_ev:]
        st = next((e for e in ev if e["type"] == "start"), None)
        fa = next((e for e in ev if e["type"] == "first_audio"), None)
        fi = next((e for e in ev if e["type"] == "final"), None)
        sev = [e for e in dev.snapshot()["events"][n_sev:] if e["t"] >= t_dac0]
        lv = [pk for (t, pk) in list(dev.levels) if t_dac0 <= t <= t_end + 0.3]
        tr = {"i": k, "file": os.path.basename(it["path"]), "set": it["set"], "label": it["label"],
              "max_event_peak": max([e["peak"] for e in sev], default=0.0), "rx_peak_dbfs": max(lv, default=None),
              "detected": bool(det), "n_detections": len(det), "score": det[0]["score"] if det else None,
              "kw_end_s": it["kw_end"], "clip_s": round(len(x) / 16000, 2), "transcript": fi.get("text") if fi else None,
              "ref_transcript": it["ref"]}
        if det and it["kw_end"] is not None:
            t_kw = t_dac0 + it["kw_end"] + ACOUSTIC_S
            tr["serial_detect_after_kw_ms"] = round((det[0]["t"] - t_kw) * 1000, 1)  # biased late by UART (~8-20 ms)
            if st:
                tr["server_start_after_kw_ms"] = round((st["t"] - t_kw) * 1000, 1)
            if fa:
                tr["latency_ms"] = round((fa["t"] - t_kw) * 1000, 1)
        trials.append(tr)
        print(f"[{k + 1}/{len(items)}] {tr['file']} label={it['label']} det={len(det)} "
              f"lat={tr.get('latency_ms')} text={tr['transcript']!r}", flush=True)

    time.sleep(12)  # one more status line
    snap = dev.snapshot()
    dev.close()
    # device-side numbers per stream (board clock): detection -> first audio sent
    dev_first = [f["ms"] for f in snap["first_audio"]]
    pos = [t for t in trials if t["label"] == 1]
    neg = [t for t in trials if t["label"] == 0]
    lat = [t["latency_ms"] for t in trials if t.get("latency_ms") is not None]
    one_way = rtt["median"] / 2 if rtt else None
    dev_med = stt.median(dev_first) if dev_first else None
    summary = {
        "evidence": "DEVICE-ACOUSTIC (recordings replayed through PC speakers)",
        "positives": len(pos), "detected": sum(t["detected"] for t in pos),
        "tpr": round(sum(t["detected"] for t in pos) / len(pos), 4) if pos else None,
        "negatives": len(neg), "false_accepts": sum(t["detected"] for t in neg),
        "false_accept_rate": round(sum(t["detected"] for t in neg) / len(neg), 4) if neg else None,
        "negative_audio_s": round(sum(t["clip_s"] for t in neg), 1),
        "latency_ms": {"n": len(lat), "median": pct(lat, 0.5), "p95": pct(lat, 0.95), "min": min(lat, default=None),
                       "max": max(lat, default=None)},
        "decomposition_ms": {
            "device_detect_to_first_send_median": dev_med,
            "network_one_way_est": one_way,
            "detection_delay_est_median": round(pct(lat, 0.5) - dev_med - one_way, 1) if lat and dev_med is not None and one_way is not None else None,
            "serial_detect_after_kw_median(biased +UART)": pct([t["serial_detect_after_kw_ms"] for t in trials if "serial_detect_after_kw_ms" in t], 0.5),
        },
        "ping_rtt_ms": rtt, "board_ip": ip,
        "transcripts_nonempty": sum(bool(t["transcript"]) for t in pos if t["detected"]),
        "status": summarize_status(snap["status"], skip_s=0),
    }
    json.dump({"label": a.label, "gain": a.gain, "level_db": a.level_db, "offline": a.offline, "summary": summary, "trials": trials, "device_first_audio_ms": dev_first},
              open(base + ".json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
