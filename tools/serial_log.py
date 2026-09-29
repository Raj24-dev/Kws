#!/usr/bin/env python3
"""Prints the ESP32's serial output and saves it (with PC timestamps) to logs/serial_<date>_<time>.log,
so long test runs can be checked later (status lines, detections, streaming, errors).

    python serial_log.py COM6                 # until Ctrl+C
    python serial_log.py COM6 --reset         # reboot the board first, to capture the boot log
    python serial_log.py COM6 --minutes 30    # stability run

Needs pyserial (already in the ESP-IDF Python environment).
"""
import argparse
import datetime
import os
import re
import time

import serial

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("--baud", type=int, default=115200)
ap.add_argument("--reset", action="store_true", help="reboot the board first (captures the boot log)")
ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = until Ctrl+C)")
ap.add_argument("--logdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"))
args = ap.parse_args()

ANSI = re.compile(r"\x1b\[[0-9;]*m")
os.makedirs(args.logdir, exist_ok=True)
path = os.path.abspath(os.path.join(args.logdir, datetime.datetime.now().strftime("serial_%Y%m%d_%H%M%S.log")))

ser = serial.Serial()
ser.port, ser.baudrate, ser.timeout = args.port, args.baud, 0.5
ser.dtr = ser.rts = False  # opening the port must not hold the chip in reset or in download mode
ser.open()
if args.reset:  # RTS pulls EN low, same as pressing RST
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False

end = time.time() + args.minutes * 60 if args.minutes else float("inf")
print(f"logging {args.port} to {path}", flush=True)
with open(path, "a", encoding="utf-8", buffering=1) as log:
    try:
        while time.time() < end:
            raw = ser.readline()
            if not raw:
                continue
            text = ANSI.sub("", raw.decode("utf-8", "replace").rstrip())
            line = f"{datetime.datetime.now():%H:%M:%S.%f}"[:-3] + " " + text
            if not text.startswith("@"):  # "@T"/"@I" telemetry (10/s) goes to the log file only; see dashboard.py
                print(line, flush=True)
            log.write(line + "\n")
    except KeyboardInterrupt:
        pass
