#!/usr/bin/env python3
"""Live dashboard for the KWS board: reads the serial port and serves a web page with live graphs of the
wake-word score, CPU and RAM (with lowest/highest), the microphone level, and a tester's scorecard.

    dashboard.cmd                            # Windows: double-click (finds the board by itself)
    python dashboard.py                      # finds the board's USB port, opens http://localhost:8090
    python dashboard.py COM6 --reset         # fixed port; reboot the board first
    python dashboard.py COM6 --host 0.0.0.0  # also reachable from other devices (phone, second laptop)
    python dashboard.py socket://host:4000   # any pyserial URL works (ser2net, rfc2217://...)
    python dashboard.py --selftest

Testing: press Space on the page each time you say the wake word. A detection from 1 s before to 2.5 s after
the mark counts as correct; a mark without one is a miss; a detection nobody marked is a false alarm (click
it to override). The raw serial log and the tester's marks are saved to logs/serial_<date>_<time>.log, the
scorecard via "Export CSV".

Needs pyserial (in the ESP-IDF Python environment) and firmware with KWS_TELEMETRY_MS > 0 (e.g. 100; it is 0 = off
in sdkconfig.defaults).
While the page runs the dashboard owns the COM port: use "Release port" on the page before flashing.
"""
import argparse
import csv
import datetime
import io
import json
import os
import queue
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH_BEFORE, MATCH_AFTER = 1.0, 2.5  # a detection this many seconds before/after the tester's mark belongs to it
HISTORY_S = 600                       # telemetry kept for the graphs (sent to every page that connects)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
KV = re.compile(r"(\w+)=(\S+)")
DETECT = re.compile(r'>>> WAKE WORD "(.*?)" DETECTED\s+\(score ([\d.]+)')
SCORE_EVENT = re.compile(r"score event: peak ([\d.]+) over (\d+) ms -> not detected")
BOOT = re.compile(r"^rst:0x\w+ \((\w+)\)")
CRASH = re.compile(r"Guru Meditation|abort\(\) was called|Stack protection fault|Task watchdog")


def evaluate(dets, marks, now):
    """Pairs the tester's marks with detections (both sorted by time T, seconds).
    dets: [{id, T, ov}] where ov is the tester's override "tp"/"fp"/None. marks: [{id, T}].
    Returns ({det id: (verdict, mark id)}, {mark id: (result, det id)}), verdict tp/fp/pending, result hit/miss/pending.
    """
    dv, mr, used = {}, {}, set()
    for m in sorted(marks, key=lambda m: m["T"]):
        d = next((d for d in dets if d["id"] not in used and d["ov"] != "fp"
                  and m["T"] - MATCH_BEFORE <= d["T"] <= m["T"] + MATCH_AFTER), None)
        if d:
            used.add(d["id"])
            mr[m["id"]] = ("hit", d["id"])
            dv[d["id"]] = ("tp", m["id"])
        else:
            mr[m["id"]] = ("pending" if now < m["T"] + MATCH_AFTER else "miss", None)
    for d in dets:
        if d["ov"]:
            dv[d["id"]] = (d["ov"], dv.get(d["id"], (None, None))[1])
        elif d["id"] not in dv:  # a mark pressed up to MATCH_BEFORE late can still claim it
            dv[d["id"]] = ("pending" if now < d["T"] + MATCH_BEFORE else "fp", None)
    return dv, mr


def ratio(a, b):
    return round(a / b, 4) if b else None


class Session:
    def __init__(self, port, log_path, echo=True):
        self.lock = threading.Lock()
        self.clients = []
        self.echo = echo
        self.serial = {"port": port, "status": "connecting", "msg": "", "want_open": True}
        self.info = {}
        self.tel = []      # telemetry samples of the last HISTORY_S seconds
        self.logs = []     # last 300 non-telemetry lines
        self.last_t = self.last_T = None
        self.boot_reason = self.crash = ""
        self.log_path = log_path
        self.log_file = open(log_path, "a", encoding="utf-8", buffering=1) if log_path else None
        self.reset()

    def reset(self):
        self.start = time.time()
        self.dets, self.marks, self.near, self.boots = [], [], [], []
        self.ext = {}
        self.listen_s = 0.0
        self.next_id = 1

    # --- output -----------------------------------------------------------------------------------------------
    def write_log(self, T, text):
        if self.log_file:
            self.log_file.write(f"{datetime.datetime.fromtimestamp(T):%H:%M:%S.%f}"[:-3] + " " + text + "\n")

    def broadcast(self, obj):
        data = json.dumps(obj, separators=(",", ":"))
        for q in list(self.clients):
            try:
                q.put_nowait(data)
            except queue.Full:  # a page that stopped reading: drop it (EventSource reconnects if it is alive)
                self.clients.remove(q)

    def subscribe(self):
        q = queue.Queue(maxsize=5000)
        with self.lock:
            q.put(json.dumps({**self.state(), "type": "snap", "tel": self.tel, "logs": self.logs, "ext": self.ext},
                             separators=(",", ":")))
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def state(self):
        now = time.time()
        dv, mr = evaluate(self.dets, self.marks, now)
        dets = [{**d, "v": dv[d["id"]][0], "mark": dv[d["id"]][1]} for d in self.dets]
        marks = []
        for m in self.marks:
            r, det = mr[m["id"]]
            near = None
            if r == "miss":  # was it close? the best near-miss inside the mark's window
                peaks = [n["peak"] for n in self.near if m["T"] - MATCH_BEFORE <= n["T"] <= m["T"] + MATCH_AFTER]
                near = max(peaks) if peaks else None
            marks.append({**m, "r": r, "det": det, "near": near})
        verdicts = [v for v, _ in dv.values()]
        tp, fp = verdicts.count("tp"), verdicts.count("fp")
        fn = sum(r == "miss" for r, _ in mr.values())
        pending = verdicts.count("pending") + sum(r == "pending" for r, _ in mr.values())
        metrics = {"said": len(self.marks), "tp": tp, "fp": fp, "fn": fn, "pending": pending,
                   "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                   "fa_per_h": round(fp / (self.listen_s / 3600), 2) if self.listen_s >= 60 else None,
                   "near": len(self.near), "listen_s": round(self.listen_s)}
        return {"type": "state", "now": now, "start": self.start, "serial": self.serial, "info": self.info,
                "dets": dets, "marks": marks, "near": self.near, "boots": self.boots, "m": metrics,
                "window": [MATCH_BEFORE, MATCH_AFTER]}

    def push_state(self):
        with self.lock:
            self.broadcast(self.state())

    def set_serial(self, status, msg=""):
        with self.lock:
            self.serial.update(status=status, msg=msg)
            self.broadcast(self.state())

    # --- serial input -----------------------------------------------------------------------------------------
    def handle_line(self, raw, T=None):
        T = T or time.time()
        line = ANSI.sub("", raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw).rstrip()
        if not line:
            return
        with self.lock:
            self.write_log(T, line)
            if line.startswith("@T "):
                self.on_telemetry({k: float(v) for k, v in KV.findall(line)}, T)
                return
            if line.startswith("@I "):
                info = {k: (float(v) if re.fullmatch(r"-?[\d.]+", v) else v) for k, v in KV.findall(line)}
                if info != self.info:
                    self.info = info
                    self.broadcast(self.state())
                return
            if self.echo:
                print(f"{datetime.datetime.fromtimestamp(T):%H:%M:%S.%f}"[:-3], line, flush=True)
            changed = True
            if m := DETECT.search(line):
                self.dets.append({"id": self.new_id(), "T": round(T, 3), "score": float(m.group(2)), "ov": None})
            elif m := SCORE_EVENT.search(line):  # a rise of the score that did not fire
                start = T - int(m.group(2)) / 1000 - 0.5  # the line comes at the end of the event
                best = max((x for x in self.tel if x["T"] >= start), key=lambda x: x["s"], default=None)
                self.near.append({"id": self.new_id(), "T": best["T"] if best else round(T, 3),
                                  "peak": float(m.group(1)), "ms": int(m.group(2))})
            else:
                changed = False
                if m := BOOT.search(line):
                    self.boot_reason = m.group(1)
                elif CRASH.search(line):
                    self.crash = line.strip()[:120]
            self.logs = (self.logs + [{"T": round(T, 3), "text": line}])[-300:]
            self.broadcast({"type": "log", "T": round(T, 3), "text": line})
            if changed:
                self.broadcast(self.state())

    def new_id(self):
        self.next_id += 1
        return self.next_id - 1

    def on_telemetry(self, v, T):
        if "t" not in v or "ht" not in v:
            return
        if self.last_t is not None and v["t"] < self.last_t - 1000:  # device clock went back: it rebooted
            self.boots.append({"T": round(T, 3), "why": self.crash or self.boot_reason or "reset"})
            self.boot_reason = self.crash = ""
            self.broadcast(self.state())
        self.last_t = v["t"]
        if self.last_T is not None and T - self.last_T < 1.5:  # gaps (unplugged, port released) are not listening
            self.listen_s += T - self.last_T
        self.last_T = T

        c0, c1 = v.get("c0", -1), v.get("c1", -1)
        x = {"T": round(T, 3), "t": int(v["t"]), "s": v.get("s", 0), "lv": v.get("lv"), "pk": v.get("pk"),
             "c0": c0 if c0 >= 0 else None, "c1": c1 if c1 >= 0 else None,
             "c": round((c0 + c1) / 2, 1) if c0 >= 0 and c1 >= 0 else None,
             "kw": v.get("kw"), "inf": v.get("inf") if v.get("n") else None, "n": v.get("n"), "d": v.get("d"),
             "ram": v["ht"] - v["hf"], "ramT": v["ht"], "ramP": v["ht"] - v["hm"],
             "ps": v.get("pt", 0) - v.get("pf", 0), "psT": v.get("pt", 0), "psP": v.get("pt", 0) - v.get("pm", 0)}
        for k in ("c", "c0", "c1", "ram", "ps", "inf", "s", "lv"):  # lowest / highest since start or reset
            if x[k] is not None:
                lo, hi = self.ext.get(k, (x[k], x[k]))
                self.ext[k] = (min(lo, x[k]), max(hi, x[k]))
        self.tel.append(x)
        if self.tel[0]["T"] < T - HISTORY_S - 10:
            self.tel = [y for y in self.tel if y["T"] >= T - HISTORY_S]
        self.broadcast({**x, "type": "tel", "ext": self.ext})

    # --- tester actions (from the page) -----------------------------------------------------------------------
    def action(self, kind, body):
        T = time.time()
        with self.lock:
            if kind == "mark":
                self.marks.append({"id": self.new_id(), "T": round(T, 3)})
                self.write_log(T, f"## tester: said the wake word (mark {self.next_id - 1})")
            elif kind == "verdict":
                for d in self.dets:
                    if d["id"] == body.get("id"):
                        d["ov"] = body.get("v") if body.get("v") in ("tp", "fp") else None
                        self.write_log(T, f"## tester: detection {d['id']} -> {d['ov'] or 'automatic'}")
            elif kind == "unmark":
                self.marks = [m for m in self.marks if m["id"] != body.get("id")]
                self.write_log(T, f"## tester: removed mark {body.get('id')}")
            elif kind == "reset":
                self.write_log(T, "## tester: reset (new test run)")
                self.reset()
            elif kind == "serial":
                self.serial["want_open"] = bool(body.get("open"))
            else:
                return False
            self.broadcast(self.state())
            if kind == "reset":
                self.broadcast({"type": "reset", "ext": self.ext})
            return True

    def export_csv(self):
        with self.lock:
            st = self.state()
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        ts = lambda T: f"{datetime.datetime.fromtimestamp(T):%Y-%m-%d %H:%M:%S.%f}"[:-3]
        m = st["m"]
        w.writerow(["# KWS test run", ts(st["start"]), "wake word", st["info"].get("ww", "?"),
                    "threshold", st["info"].get("cut", "?")])
        w.writerow(["# said", m["said"], "correct", m["tp"], "false alarms", m["fp"], "missed", m["fn"],
                    "precision", m["precision"], "recall", m["recall"], "false alarms/h", m["fa_per_h"],
                    "listening s", m["listen_s"], "near misses", m["near"]])
        w.writerow(["time", "event", "score", "result", "detail"])
        rows = []
        for d in st["dets"]:
            how = "manual" if d["ov"] else (f"mark {d['mark']}" if d["mark"] else "")
            rows.append((d["T"], "detection", d["score"], d["v"], how))
        for k in st["marks"]:
            detail = f"detection {k['det']}" if k["det"] else (f"near miss peak {k['near']}" if k["near"] else "")
            rows.append((k["T"], "said", "", k["r"], detail))
        for n in st["near"]:
            rows.append((n["T"], "near miss", n["peak"], "below threshold or cool-down", f"{n['ms']} ms"))
        for b in st["boots"]:
            rows.append((b["T"], "reboot", "", b["why"], ""))
        for T, *rest in sorted(rows):
            w.writerow([ts(T), *rest])
        return out.getvalue()


# --- serial thread --------------------------------------------------------------------------------------------
USB_SERIAL_VIDS = (0x303A, 0x1A86, 0x10C4, 0x0403)  # Espressif USB, WCH CH34x, SiLabs CP210x, FTDI


def find_board():
    """The first USB serial port with a known ESP32 dev-board USB chip (None if the board is unplugged)."""
    from serial.tools import list_ports
    ports = sorted((p for p in list_ports.comports() if p.vid in USB_SERIAL_VIDS),
                   key=lambda p: USB_SERIAL_VIDS.index(p.vid))
    return ports[0].device if ports else None


def serial_reader(S, port, baud, reset):
    import serial  # pyserial

    while True:
        if not S.serial["want_open"]:
            if S.serial["status"] != "released":
                S.set_serial("released", "port is free (flash now, then press Reconnect)")
            time.sleep(0.3)
            continue
        name = port or find_board()  # auto: look again every time, the board may be replugged or re-enumerated
        if not name:
            S.set_serial("error", "no ESP32 board found on USB - plug it in (retrying)")
            time.sleep(2)
            continue
        S.serial["port"] = name
        try:
            ser = serial.serial_for_url(name, baudrate=baud, timeout=0.3, do_not_open=True)
            ser.dtr = ser.rts = False  # opening the port must not hold the chip in reset or in download mode
            ser.open()
        except (serial.SerialException, OSError, ValueError) as e:
            S.set_serial("error", f"{e}".split(":")[0] + " (busy? close idf.py monitor / serial_log.py; retrying)")
            time.sleep(2)
            continue
        S.set_serial("open")
        try:
            if reset:  # RTS pulls EN low, same as pressing RST
                reset = False
                ser.rts = True
                time.sleep(0.1)
                ser.rts = False
            while S.serial["want_open"]:
                raw = ser.readline()
                if raw:
                    S.handle_line(raw)
        except (serial.SerialException, OSError) as e:
            S.set_serial("error", f"connection lost: {e}"[:160])
            time.sleep(1)
        finally:
            ser.close()


def ticker(S):
    while True:  # pending verdicts resolve with time, and the listening time moves: refresh every second
        time.sleep(1)
        S.push_state()


# --- web server -----------------------------------------------------------------------------------------------
def make_handler(S):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]  # the pages take ?asr=ws://host:port/live
            if path in ("/", "/index.html", "/transcription"):
                page = "transcription.html" if path == "/transcription" else "dashboard.html"
                with open(os.path.join(HERE, page), "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/export.csv":
                name = f"kws_run_{datetime.datetime.fromtimestamp(S.start):%Y%m%d_%H%M%S}.csv"
                self.send(200, S.export_csv().encode("utf-8-sig"), "text/csv; charset=utf-8",
                          [("Content-Disposition", f'attachment; filename="{name}"')])
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                q = S.subscribe()
                try:
                    while True:
                        try:
                            self.wfile.write(b"data: " + q.get(timeout=15).encode() + b"\n\n")
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except OSError:  # page closed
                    pass
                finally:
                    S.unsubscribe(q)
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n <= 10000 else {}
            except ValueError:
                body = {}
            ok = isinstance(body, dict) and S.action(self.path.strip("/"), body)
            self.send(200 if ok else 400, b'{"ok":%s}' % (b"true" if ok else b"false"), "application/json")

    return Handler


def selftest():
    dets = [dict(id=1, T=10.5, ov=None), dict(id=2, T=20.0, ov=None), dict(id=3, T=30.0, ov="tp"),
            dict(id=4, T=41.0, ov="fp")]
    marks = [dict(id=10, T=10.0), dict(id=11, T=40.0), dict(id=12, T=99.0)]
    dv, mr = evaluate(dets, marks, now=100.0)
    assert dv[1] == ("tp", 10) and dv[2] == ("fp", None) and dv[3][0] == "tp" and dv[4][0] == "fp", dv
    assert mr[10] == ("hit", 1) and mr[11] == ("miss", None) and mr[12] == ("pending", None), mr
    assert evaluate(dets, marks, now=20.5)[0][2] == ("pending", None)  # the tester may still press Space
    # two quick utterances pair in order
    dv, mr = evaluate([dict(id=1, T=5.8, ov=None), dict(id=2, T=6.9, ov=None)], [dict(id=8, T=5.0), dict(id=9, T=6.0)], 99)
    assert mr[8] == ("hit", 1) and mr[9] == ("hit", 2), mr

    S = Session("TEST", None, echo=False)
    S.handle_line(b"@I ww=marvin cut=0.50 win=5 cool=1000 tick=100 wifi=1 srv=0 li=0 ln=0", T=1000.0)
    for i, s in enumerate((0.01, 0.3, 0.7, 0.2, 0.01)):
        S.handle_line(f"@T t={5000 + i * 100} s={s} lv=-40.0 pk=-20.0 c0=30.0 c1=10.0 kw=5.0 inf=1.8 n=3 d=0 "
                      f"hf=200000 hm=190000 ht=400000 pf=8000000 pm=7900000 pt=8388608", T=1000.1 + i * 0.1)
    S.handle_line(b'\x1b[0;32mI (5300) kws: >>> WAKE WORD "marvin" DETECTED  (score 0.62, t = 5.30 s)\x1b[0m', T=1000.35)
    S.handle_line(b"I (9000) wake_word: score event: peak 0.43 over 300 ms -> not detected (below threshold)", T=1004.0)
    S.handle_line("@T t=2000 s=0 lv=-40 pk=-20 c0=50 c1=50 kw=5 inf=1.8 n=3 d=0 hf=100000 hm=90000 ht=400000 pf=0 pm=0 pt=0", T=1010.0)
    st = S.state()
    assert st["info"]["ww"] == "marvin" and st["info"]["cut"] == 0.5
    assert len(st["dets"]) == 1 and st["dets"][0]["score"] == 0.62 and st["dets"][0]["v"] == "fp"
    assert len(st["near"]) == 1 and st["near"][0]["peak"] == 0.43
    assert len(st["boots"]) == 1, st["boots"]
    assert S.ext["c"] == (20.0, 50.0) and S.ext["ram"] == (200000, 300000), S.ext
    assert abs(S.listen_s - 0.4) < 1e-6  # the 5.6 s gap before the reboot line is not listening time
    assert "detection" in S.export_csv()
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="COM6, /dev/ttyUSB0 or a pyserial URL (default: find the board)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--reset", action="store_true", help="reboot the board first (captures the boot log)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from other devices on the network")
    ap.add_argument("--http-port", type=int, default=8090)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--logdir", default=os.path.join(HERE, "..", "logs"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(args.logdir, datetime.datetime.now().strftime("serial_%Y%m%d_%H%M%S.log")))
    S = Session(args.port or "auto", log_path)
    threading.Thread(target=serial_reader, args=(S, args.port, args.baud, args.reset), daemon=True).start()
    threading.Thread(target=ticker, args=(S,), daemon=True).start()
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.http_port}"
    try:
        server = ThreadingHTTPServer((args.host, args.http_port), make_handler(S))
    except OSError:  # port taken: most likely the dashboard is already running, so just show it
        print(f"port {args.http_port} is in use - the dashboard is probably already running: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return
    server.daemon_threads = True
    print(f"dashboard: {url}   serial: {args.port or 'auto (finds the board)'}   log: {log_path}   (Ctrl+C to stop)",
          flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped; log saved to {log_path}")


if __name__ == "__main__":
    main()
