"""Idle-listening CPU + RAM on the board, Wi-Fi and server connected, nothing streamed.  [DEVICE-ACOUSTIC]

    python run_idle.py --label baseline --cond quiet  --minutes 6
    python run_idle.py --label baseline --cond speech --minutes 6   # PC speakers loop synthetic speech (no wake word)
                                                                   # so the model never pauses: worst case

Resets the board, waits until the status line shows "wifi OK, server OK", discards the first 60 s, then averages the
firmware's 10 s status windows (CPU = 100 % - FreeRTOS idle-task share, per core). Writes results/<label>_idle_<cond>.json
and the raw serial log next to it.
"""
import argparse
import json
import os
import time

from devlog import DeviceLog, summarize_status

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM6")
ap.add_argument("--label", required=True)
ap.add_argument("--cond", choices=["quiet", "speech"], default="quiet")
ap.add_argument("--minutes", type=float, default=6)
ap.add_argument("--gain", type=float, default=0.5, help="playback gain for --cond speech")
ap.add_argument("--no-reset", action="store_true")
a = ap.parse_args()

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
base = os.path.join(HERE, "results", f"{a.label}_idle_{a.cond}")
dev = DeviceLog(a.port, base + ".log", reset=not a.no_reset)
t0 = time.time()
while time.time() - t0 < 90:  # boot + Wi-Fi + WebSocket
    st = dev.snapshot()["status"]
    if st and st[-1]["srv"] == "OK":
        break
    time.sleep(1)
else:
    print("WARNING: server connection not seen within 90 s (measuring anyway)")
up0 = dev.snapshot()["status"][-1]["up"] if dev.snapshot()["status"] else 0
looper = None
if a.cond == "speech":
    from playback import Looper, load16k
    looper = Looper(load16k(os.path.join(HERE, "audio", "synthetic_idle_speech.wav")), a.gain)
print(f"measuring {a.minutes} min after a 60 s warm-up ...", flush=True)
time.sleep(60 + a.minutes * 60 + 5)
if looper:
    looper.stop()
dev.close()

snap = dev.snapshot()
res = {"label": a.label, "condition": a.cond, "evidence": "DEVICE-ACOUSTIC",
       "audio": "synthetic TTS speech looped from the PC speakers" if a.cond == "speech" else "room ambient only",
       "boot": dev.banner, "heap_regions": dev.heap_regions,
       "summary": summarize_status(snap["status"], skip_s=up0 + 60),
       "detections": snap["detections"]}
json.dump(res, open(base + ".json", "w"), indent=1, default=str)
s = res["summary"]
print(json.dumps({k: s[k] for k in ("minutes", "cpu_core0", "cpu_core1", "cpu_sum_both_cores", "ram_used_kb",
                                    "ram_peak_since_boot_kb", "server_ok_share", "detections_during")}, indent=1))
