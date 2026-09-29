"""Serial capture + parser for the kws_s3 firmware's log (shared by the benchmark scripts).

DeviceLog opens the board's serial port in a background thread, stamps every line with time.perf_counter()
and the wall clock, writes the raw log to a file, and parses the lines the benchmarks need:
  [status] ...            every 10 s: CPU per core (FreeRTOS idle-task share), RAM used/peak, detections
  >>> WAKE WORD ... DETECTED (score s, t = x s)
  first audio sent N ms after the detection / stream finished (...)
  heap_init / banner lines (static RAM, IRAM code, heap regions)
"""
import re
import threading
import time
from datetime import datetime

import serial

STATUS = re.compile(
    r"\[status\] up (?P<up>\d+)s \| mic\s+(?P<mic>-?[\d.]+) dBFS \(peak\s+(?P<mic_pk>-?[\d.]+)\) \| score max (?P<smax>[\d.]+)"
    r" \| detections (?P<det>\d+) \| inferences (?P<inf>\d+) \([\d.]+/s; model (?P<model_ms>[\d.]+) ms each, paused in quiet"
    r" (?P<paused>[\d.]+)%; features (?P<feat_ms>[\d.]+) ms per 30 ms\) \| CPU core0\s+(?P<c0>[\d.-]+)% core1\s+(?P<c1>[\d.-]+)%"
    r" \(wake word pipeline\s+(?P<pipe>[\d.]+)% of one core\) \| RAM used (?P<ram>\d+) KB \(peak (?P<ram_pk>\d+) KB"
    r"(?:; peak \+ IRAM code (?P<strict_pk>\d+) KB)?(?: of the \d+ KB limit)?, (?P<free>\d+) KB free\)"
    r"(?: \| wifi (?P<wifi>\S+), server (?P<srv>\S+))?"
    r"(?: \| mics: mix left (?P<mix_left>\d+)% right \d+% \(right noise (?P<mic_noise_db>[+-]?[\d.]+) dB\)"
    r"(?:, right (?P<mic_delay_us>[+-]?\d+) us after left[^|]*)?)?"
    r" \| audio lost since boot: i2s (?P<li2s>\d+) net (?P<lnet>\d+)")
DETECT = re.compile(r">>> WAKE WORD \"(?P<ww>[^\"]+)\" DETECTED\s+\(score (?P<score>[\d.]+), t = (?P<t>[\d.]+) s\)")
FIRST = re.compile(r"first audio sent (?P<ms>-?\d+) ms after the detection")
FINISHED = re.compile(r"stream finished \((?P<reason>[^)]+)\): (?P<ms>\d+) ms of audio sent")
BANNER_RAM = re.compile(r"RAM\s+: (?P<static>\d+) KB static data, (?P<heapfree>\d+) KB heap free \(code in IRAM: (?P<iram>\d+) KB")
HEAP_REGION = re.compile(r"heap_init: At (?P<addr>[0-9A-F]+) len (?P<len>[0-9A-F]+) \((?P<kib>\d+) KiB\): (?P<kind>\w+)")
EVENT = re.compile(r"score event: peak (?P<peak>[\d.]+) over (?P<ms>\d+) ms -> (?P<res>\w+)")
TELE_PK = re.compile(r"^@T .*? pk=(?P<pk>-?[\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class DeviceLog:
    def __init__(self, port, logfile, baud=115200, reset=False):
        self.lines = []            # (perf_counter, wall datetime, text)
        self.status, self.detections, self.first_audio, self.finished = [], [], [], []
        self.events, self.levels = [], []  # score events (peak of every rise > 0.20), 100 ms mic peaks
        self.banner, self.heap_regions = {}, []
        self._lock = threading.Lock()
        self._stop = False
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, baud, 0.2
        self.ser.dtr = self.ser.rts = False  # opening the port must not hold the chip in reset / download mode
        self.ser.open()
        self.log = open(logfile, "a", encoding="utf-8", buffering=1)
        if reset:  # RTS pulls EN low = RST button
            self.ser.rts = True
            time.sleep(0.1)
            self.ser.rts = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        buf = b""
        while not self._stop:
            buf += self.ser.read(4096)
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                t = time.perf_counter()
                now = datetime.now()
                text = ANSI.sub("", raw.decode("utf-8", "replace").rstrip())
                self.log.write(f"{now:%H:%M:%S.%f}"[:-3] + " " + text + "\n")
                self._parse(t, now, text)

    def _parse(self, t, now, text):
        with self._lock:
            self.lines.append((t, now, text))
            if text.startswith("@T"):
                if m := TELE_PK.search(text):
                    self.levels.append((t, float(m["pk"])))
                return
            if m := STATUS.search(text):
                d = {k: (float(v) if k not in ("wifi", "srv") and v is not None else v) for k, v in m.groupdict().items()}
                d["t"] = t
                self.status.append(d)
            elif m := DETECT.search(text):
                self.detections.append({"t": t, "score": float(m["score"]), "dev_t": float(m["t"])})
            elif m := FIRST.search(text):
                self.first_audio.append({"t": t, "ms": int(m["ms"])})
            elif m := FINISHED.search(text):
                self.finished.append({"t": t, "reason": m["reason"], "ms": int(m["ms"])})
            elif m := EVENT.search(text):
                self.events.append({"t": t, "peak": float(m["peak"]), "ms": int(m["ms"]), "detected": m["res"] == "detected"})
            elif m := BANNER_RAM.search(text):
                self.banner = {k: int(v) for k, v in m.groupdict().items()}
            elif m := HEAP_REGION.search(text):
                self.heap_regions.append({"addr": m["addr"], "bytes": int(m["len"], 16), "kind": m["kind"]})

    def snapshot(self):
        with self._lock:
            return {"status": list(self.status), "detections": list(self.detections),
                    "first_audio": list(self.first_audio), "finished": list(self.finished),
                    "events": list(self.events)}

    def wait_for(self, pattern, timeout):
        """Waits until a line matching pattern arrives (searching lines received from now on)."""
        rx = re.compile(pattern)
        start = len(self.lines)
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                for _, _, text in self.lines[start:]:
                    if rx.search(text):
                        return text
                start = len(self.lines)
            time.sleep(0.05)
        return None

    def close(self):
        self._stop = True
        self.thread.join(timeout=2)
        self.ser.close()
        self.log.close()


def parse_file(path):
    """Re-parses a saved log (the time stamps become seconds since midnight) - for re-analysis without the board."""
    d = DeviceLog.__new__(DeviceLog)
    d.lines, d.status, d.detections, d.first_audio, d.finished, d.banner, d.heap_regions = [], [], [], [], [], {}, []
    d.events, d.levels = [], []
    d._lock = threading.Lock()
    for line in open(path, encoding="utf-8", errors="replace"):
        hh, mm, ss = line[:12].split(":")
        d._parse(int(hh) * 3600 + int(mm) * 60 + float(ss), None, line[13:].rstrip())
    return d


def summarize_status(rows, skip_s=0):
    """CPU/RAM statistics over status lines (each covers the previous 10 s)."""
    import statistics as st
    rows = [r for r in rows if r["up"] > skip_s]
    if not rows:
        return {}

    def stats(key):
        v = sorted(r[key] for r in rows if not (key in ("c0", "c1") and r[key] < 0))  # -1 = no previous sample
        return {"mean": round(st.fmean(v), 2), "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 2),
                "max": round(v[-1], 2), "min": round(v[0], 2)}

    both = sorted(r["c0"] + r["c1"] for r in rows)
    return {
        "windows_10s": len(rows), "minutes": round(len(rows) * 10 / 60, 1),
        "cpu_core0": stats("c0"), "cpu_core1": stats("c1"),
        "cpu_sum_both_cores": {"mean": round(sum(both) / len(both), 2), "p95": both[min(len(both) - 1, int(0.95 * len(both)))],
                               "max": both[-1]},
        "pipeline_pct_one_core": stats("pipe"), "model_ms": stats("model_ms"), "features_ms_per_30ms": stats("feat_ms"),
        "paused_in_quiet_pct": stats("paused"), "mic_dbfs": stats("mic"),
        "ram_used_kb": stats("ram"), "ram_peak_since_boot_kb": max(r["ram_pk"] for r in rows),
        # strict SIH reading (IRAM code + static data + peak heap); firmware from F8 on prints it
        "ram_strict_peak_kb": max((r["strict_pk"] for r in rows if r["strict_pk"] is not None), default=None),
        "heap_free_kb_min": min(r["free"] for r in rows),
        "wifi_ok_share": round(sum(r["wifi"] == "OK" for r in rows) / len(rows), 3),
        "server_ok_share": round(sum(r["srv"] == "OK" for r in rows) / len(rows), 3),
        "detections_during": int(rows[-1]["det"] - rows[0]["det"]),
        "audio_lost_i2s": int(rows[-1]["li2s"]), "audio_lost_net": int(rows[-1]["lnet"]),
    }


if __name__ == "__main__":  # self-check of the parser on real lines from the firmware
    d = DeviceLog.__new__(DeviceLog)
    d.lines, d.status, d.detections, d.first_audio, d.finished, d.banner, d.heap_regions = [], [], [], [], [], {}, []
    d.events, d.levels = [], []
    d._lock = threading.Lock()
    d._parse(1.0, datetime.now(), "[status] up 8089s | mic -32.6 dBFS (peak -19.2) | score max 0.02 | detections 12 | inferences 318 (31.8/s; model 1.17 ms each, paused in quiet 6%; features 1.42 ms per 30 ms) | CPU core0  5.4% core1  9.5% (wake word pipeline 9.24% of one core) | RAM used 230 KB (peak 245 KB, 93 KB free) | wifi OK, server OK | audio lost since boot: i2s 0 net 0")
    d._parse(2.0, datetime.now(), 'I (123) kws: >>> WAKE WORD "marvin" DETECTED  (score 0.58, t = 12.34 s)')
    d._parse(3.0, datetime.now(), " RAM       : 46 KB static data, 248 KB heap free (code in IRAM: 95 KB, not counted)")
    d._parse(4.0, datetime.now(), "I (8053747) wake_word: score event: peak 0.23 over 390 ms -> not detected (below threshold or in cool-down)")
    d._parse(5.0, datetime.now(), "@T t=182852 s=0.000 lv=-34.2 pk=-33.7 c0=4.2 c1=10.3")
    assert d.events[0]["peak"] == 0.23 and not d.events[0]["detected"] and d.levels[0][1] == -33.7
    d._parse(6.0, datetime.now(), "[status] up 21s | mic -30.1 dBFS (peak -26.0) | score max 0.00 | detections 0 | inferences 333 (33.3/s; model 0.87 ms each, paused in quiet 0%; features 1.39 ms per 30 ms) | CPU core0  0.9% core1  8.4% (wake word pipeline 8.16% of one core) | RAM used 173 KB (peak 176 KB; peak + IRAM code 230 KB of the 256 KB limit, 28 KB free) | wifi OK, server OK | audio lost since boot: i2s 0 net 0")
    assert d.status[0]["c1"] == 9.5 and d.status[0]["ram_pk"] == 245 and d.status[0]["srv"] == "OK"
    assert d.status[1]["strict_pk"] == 230 and d.status[1]["free"] == 28 and d.status[0]["strict_pk"] is None
    assert d.detections[0]["score"] == 0.58 and d.banner["iram"] == 95
    d._parse(7.0, datetime.now(), "[status] up 21s | mic -31.8 dBFS (peak -25.4) | score max 0.00 | detections 0 | inferences 0 (0.0/s; model 0.00 ms each, paused in quiet 100%; features 1.35 ms per 30 ms) | CPU core0  1.4% core1  6.1% (wake word pipeline 5.93% of one core) | RAM used 178 KB (peak 182 KB; peak + IRAM code 236 KB of the 256 KB limit, 23 KB free) | wifi OK, server OK | mics: mix left 100% right 0% (right noise +30.5 dB), right +46 us after left (talker 17 deg to the left), 2 estimates | audio lost since boot: i2s 0 net 0")
    assert d.status[2]["srv"] == "OK" and d.status[2]["mix_left"] == 100 and d.status[2]["mic_delay_us"] == 46
    assert d.status[0]["mix_left"] is None
    print("devlog parser OK")
