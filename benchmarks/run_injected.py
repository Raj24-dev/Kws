"""[DEVICE-INJECTED] accuracy: test audio goes over the serial port straight into the board's own pipeline
(features + int8 model + sliding window + cool-down + quiet-room gate, i.e. ww_process() on the ESP32-S3).
Needs the injection firmware (CONFIG_KWS_INJECT_TEST=y, see README). Each clip is fed exactly like the host mirror
and the boot self-test: 1.2 s quiet noise + clip + 0.6 s quiet noise, detector reset per clip. The same clips also run
through the PC mirror (check_model.py) so the two can be compared number for number.

    python run_injected.py --label baseline --cutoff 0.5 sets/real_test.csv audio/synthetic_confusable ...
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
from check_model import LEAD_MS, TAIL_MS, Detector, LcgNoise, load_wav, run  # noqa: E402

REC = os.path.join(ROOT, "server", "recordings")
ANS = re.compile(r"INJ (\d+) det=(\d+) first=(-?\d+) max=([\d.]+)")


def items_of(src):
    if src.endswith(".csv"):
        return [(os.path.join(REC, r["file"]), int(r["label"])) for r in csv.DictReader(open(src, encoding="utf-8"))]
    return [(p, int("marvin" in os.path.basename(p).lower() and not os.path.basename(p).startswith("neg")))
            for p in sorted(glob.glob(os.path.join(src, "*.wav")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--cutoff", type=float, default=0.5, help="the cutoff the firmware was built with (for the mirror)")
    ap.add_argument("--model", default=os.path.join(ROOT, "firmware", "model", "marvin.tflite"))
    a = ap.parse_args()

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = a.port, 115200, 0.5
    ser.dtr = ser.rts = False
    ser.open()
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False  # reset: boot straight into injection mode
    t0 = time.time()
    while time.time() - t0 < 20:
        line = ser.readline().decode("utf-8", "replace")
        if "INJECT READY" in line:
            break
    else:
        sys.exit("no 'INJECT READY' - is the injection firmware flashed?")
    time.sleep(0.3)
    ser.baudrate = 921600
    ser.reset_input_buffer()

    mirror = Detector(a.model, a.cutoff, 5)
    results = {}
    k = 0
    for src in a.sources:
        name = os.path.splitext(os.path.basename(os.path.normpath(src)))[0]
        rows = []
        for path, label in items_of(src):
            noise = LcgNoise(1)
            x = np.concatenate([noise.take(LEAD_MS * 16), load_wav(path), noise.take(TAIL_MS * 16)]).astype("<i2")
            ser.write(f"INJ {k} {len(x)}\n".encode() + x.tobytes())
            ans = None
            t1 = time.time()
            while time.time() - t1 < 60 and not ans:
                m = ANS.search(ser.readline().decode("utf-8", "replace"))
                if m and int(m.group(1)) == k:
                    ans = m
            if not ans:
                sys.exit(f"no answer for clip {k} ({path})")
            found, mx = run(mirror, [x])
            row = {"file": os.path.basename(path), "label": label, "dev_det": int(ans.group(2)),
                   "dev_first": int(ans.group(3)), "dev_max": float(ans.group(4)), "host_det": len(found),
                   "host_first": found[0][0] if found else -1, "host_max": round(float(mx), 3)}
            row["identical"] = (row["dev_det"] == row["host_det"] and row["dev_first"] == row["host_first"]
                                and abs(row["dev_max"] - row["host_max"]) < 0.0015)
            rows.append(row)
            k += 1
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        results[name] = {
            "n_pos": len(pos), "n_neg": len(neg),
            "device_tpr": round(sum(r["dev_det"] > 0 for r in pos) / len(pos), 4) if pos else None,
            "device_far": round(sum(r["dev_det"] > 0 for r in neg) / len(neg), 4) if neg else None,
            "host_tpr": round(sum(r["host_det"] > 0 for r in pos) / len(pos), 4) if pos else None,
            "host_far": round(sum(r["host_det"] > 0 for r in neg) / len(neg), 4) if neg else None,
            "device_equals_host": sum(r["identical"] for r in rows), "clips": len(rows), "rows": rows}
        s = results[name]
        print(f"{name:26s} +{s['n_pos']:3d} -{s['n_neg']:3d} | device TPR {s['device_tpr']} FAR {s['device_far']} | "
              f"host TPR {s['host_tpr']} FAR {s['host_far']} | identical {s['device_equals_host']}/{s['clips']}", flush=True)
    ser.close()
    out = os.path.join(HERE, "results", f"{a.label}_injected_c{a.cutoff}.json")
    json.dump({"label": a.label, "evidence": "DEVICE-INJECTED", "cutoff": a.cutoff, "sets": results}, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
