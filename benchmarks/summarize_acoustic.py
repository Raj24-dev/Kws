"""Pools run_acoustic.py results: latency (median/p95), detection rate, false accepts, transcript word error rate on
the synthetic latency clips (their command text is known), device-side detection -> first send, RAM peak.

    python summarize_acoustic.py results/baseline_acoustic_latency_synth_pass1.json results/..._pass2.json ...
"""
import csv
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
       "8": "eight", "9": "nine", "10": "ten"}


def words(t):
    return [NUM.get(w, w) for w in re.findall(r"[a-z0-9']+", (t or "").lower())]


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def main(paths):
    cmd = {r["file"]: r["command"] for r in csv.DictReader(open(os.path.join(HERE, "sets", "synthetic_latency.csv")))}
    lat, pos, det, neg, fa, errs, refw, first, rampk, trials = [], 0, 0, 0, 0, 0, 0, [], [], 0
    for p in paths:
        d = json.load(open(p))
        first += d.get("device_first_audio_ms", [])
        rampk.append(d["summary"]["status"].get("ram_peak_since_boot_kb"))
        for t in d["trials"]:
            trials += 1
            if t["label"] == 1:
                pos += 1
                det += t["detected"]
                if t.get("latency_ms") is not None:
                    lat.append(t["latency_ms"])
                if t["detected"] and t["file"] in cmd:
                    e, n = wer(cmd[t["file"]], t["transcript"])
                    errs += e
                    refw += n
            else:
                neg += 1
                fa += t["detected"]
    out = {"files": [os.path.basename(p) for p in paths], "trials": trials,
           "positives": pos, "detected": det, "tpr": round(det / pos, 4) if pos else None,
           "negatives": neg, "false_accepts": fa,
           "latency_ms": {"n": len(lat), "median": pct(lat, 0.5), "p95": pct(lat, 0.95), "mean": round(st.fmean(lat), 1) if lat else None,
                          "min": min(lat, default=None), "max": max(lat, default=None)},
           "device_detect_to_first_send_ms": {"median": pct(first, 0.5), "p95": pct(first, 0.95), "n": len(first)},
           "synthetic_command_wer": round(errs / refw, 4) if refw else None,
           "ram_peak_kb_per_run": rampk}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
