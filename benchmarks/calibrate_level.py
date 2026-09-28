"""Speaker-to-microphone level check: loops the idle-test speech at a given digital gain and prints the board's average
mic level (status lines). Used to keep the acoustic tests at the same received level when the PC's volume changes.

    python calibrate_level.py --gain 0.5 --seconds 45
"""
import argparse
import os
import time

from devlog import DeviceLog
from playback import Looper, load16k

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM6")
ap.add_argument("--gain", type=float, default=0.5)
ap.add_argument("--seconds", type=float, default=45)
a = ap.parse_args()

dev = DeviceLog(a.port, os.path.join(HERE, "results", "calibration.log"))
time.sleep(12)  # one status line of room noise
quiet = [s["mic"] for s in dev.snapshot()["status"]]
n0 = len(dev.snapshot()["status"])
lp = Looper(load16k(os.path.join(HERE, "audio", "synthetic_idle_speech.wav")), a.gain)
time.sleep(a.seconds)
lp.stop()
dev.close()
rows = dev.snapshot()["status"][n0 + 1:]  # skip the window that straddles the start
print(f"gain {a.gain}: room {quiet[-1] if quiet else None} dBFS; speech mic mean "
      f"{[r['mic'] for r in rows]} dBFS, peaks {[r['mic_pk'] for r in rows]}")
