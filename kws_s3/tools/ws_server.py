#!/usr/bin/env python3
"""Test server for the ESP32 wake word firmware (stands in for your cloud ASR server).

It receives the audio the ESP32 streams after the wake word, saves every utterance as a WAV file in
./recordings (so you can LISTEN to exactly what the device sent), prints timing, and can optionally
transcribe it with Whisper and send the text back (the ESP32 prints it on the serial monitor).

    pip install websockets                  (required)
    pip install faster-whisper              (optional, for --asr)
    python ws_server.py                     # listen on port 8765
    python ws_server.py --asr               # also transcribe with Whisper (downloads a small model once)

Then set the server address in the firmware (idf.py menuconfig -> KWS settings) to ws://<this PC's IP>:8765
The IP addresses of this computer are printed at startup. Allow Python through the Windows firewall
("Private networks") when Windows asks, and use the same Wi-Fi network (2.4 GHz) for PC and ESP32.
"""
import argparse
import asyncio
import datetime
import json
import math
import os
import socket
import time
import wave
from array import array

import websockets

RECORDINGS = "recordings"
ASR_MODEL = None


def local_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:  # the address used for outgoing traffic (works even when the hostname lookup does not)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def transcribe(path):
    global ASR_MODEL
    from faster_whisper import WhisperModel
    if ASR_MODEL is None:
        print("  loading Whisper 'small.en' (first time only)...")
        ASR_MODEL = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, _ = ASR_MODEL.transcribe(path, language="en", beam_size=5, temperature=0.0)  # deterministic
    return " ".join(s.text.strip() for s in segments).strip()


# G.711 mu-law byte -> int16 (same decoder as kws_s3/main/audio_input.h); the firmware streams mu-law
def _mulaw(u):
    u = ~u & 0xFF
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    return (0x84 - t) if u & 0x80 else (t - 0x84)


MULAW_TO_PCM = [array('h', [_mulaw(u)]).tobytes() for u in range(256)]


def dbfs(audio_bytes):
    """rms/peak dBFS of int16 PCM, stdlib only. -120.0 for silence/empty."""
    samples = array('h')
    samples.frombytes(bytes(audio_bytes[:len(audio_bytes) - len(audio_bytes) % 2]))
    if not samples:
        return -120.0, -120.0
    peak = max(abs(s) for s in samples)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    full_scale = 32768.0
    peak_dbfs = 20 * math.log10(peak / full_scale) if peak else -120.0
    rms_dbfs = 20 * math.log10(rms / full_scale) if rms else -120.0
    return rms_dbfs, peak_dbfs


def log_index(record):
    os.makedirs(RECORDINGS, exist_ok=True)
    with open(os.path.join(RECORDINGS, "index.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


class Utterance:
    def __init__(self, info):
        self.info = info
        self.t_start = time.perf_counter()
        self.t_first = None
        self.audio = bytearray()
        self.messages = 0

    def save(self, device):
        os.makedirs(RECORDINGS, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.join(RECORDINGS, f"{stamp}_{device}_{self.info.get('wake_word', 'utt')}.wav")
        with wave.open(name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(self.info.get("sample_rate", 16000)))
            w.writeframes(bytes(self.audio))
        return name


async def handler(ws, *_):
    peer = ws.remote_address[0] if ws.remote_address else "?"
    print(f"[+] device connected from {peer}")
    utt, device = None, peer
    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                if utt is None:
                    continue
                if utt.t_first is None:
                    utt.t_first = time.perf_counter()
                    print(f"    first audio arrived {1000 * (utt.t_first - utt.t_start):.0f} ms after the start message")
                utt.audio += b"".join(MULAW_TO_PCM[b] for b in msg) if utt.info.get("format") == "mulaw" else msg
                utt.messages += 1
                continue

            m = json.loads(msg)
            if m.get("type") == "start":
                device = m.get("device", peer)
                utt = Utterance(m)
                rtt = getattr(ws, "latency", 0) * 1000
                print(f"\n>>> START  wake word '{m.get('wake_word')}' (source {m.get('source')}, score "
                      f"{m.get('score', 0):.2f}), pre-roll {m.get('preroll_ms')} ms, network round trip ~{rtt:.0f} ms")
            elif m.get("type") == "end" and utt is not None:
                secs = len(utt.audio) / 2 / int(utt.info.get("sample_rate", 16000))
                wall = time.perf_counter() - utt.t_start
                path = utt.save(device)
                rms_dbfs, peak_dbfs = dbfs(utt.audio)
                print(f"<<< END    reason {m.get('reason')}: {secs:.2f} s of audio in {utt.messages} messages, "
                      f"{len(utt.audio) / 1024:.0f} KB ({8 * len(utt.audio) / max(secs, 1e-6) / 1000:.0f} kbit/s), "
                      f"received over {wall:.2f} s, {rms_dbfs:.1f}/{peak_dbfs:.1f} dBFS rms/peak")
                print(f"    device: detection -> first audio sent = {m.get('first_audio_latency_ms')} ms")
                print(f"    saved {path}")
                reply = {"type": "ack", "saved": os.path.basename(path), "duration_ms": int(secs * 1000)}
                text = None
                if ARGS.asr:
                    t0 = time.perf_counter()
                    text = await asyncio.to_thread(transcribe, path)
                    print(f"    transcript ({1000 * (time.perf_counter() - t0):.0f} ms): \"{text}\"")
                    reply = {"type": "transcript", "text": text}
                log_index({
                    "time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "file": os.path.basename(path),
                    "device": device,
                    "wake_word": utt.info.get("wake_word"),
                    "source": utt.info.get("source"),
                    "score": utt.info.get("score"),
                    "preroll_ms": utt.info.get("preroll_ms"),
                    "duration_s": round(secs, 2),
                    "reason": m.get("reason"),
                    "first_audio_latency_ms": m.get("first_audio_latency_ms"),
                    "rms_dbfs": rms_dbfs,
                    "peak_dbfs": peak_dbfs,
                    "transcript": text,
                })
                await ws.send(json.dumps(reply))
                utt = None
    except websockets.ConnectionClosed:
        pass
    finally:
        if utt is not None and utt.audio:
            path = utt.save(device)
            rms_dbfs, peak_dbfs = dbfs(utt.audio)
            print(f"    connection lost mid-utterance, saved {path}")
            log_index({
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "file": os.path.basename(path),
                "device": device,
                "wake_word": utt.info.get("wake_word"),
                "source": utt.info.get("source"),
                "score": utt.info.get("score"),
                "preroll_ms": utt.info.get("preroll_ms"),
                "duration_s": round(len(utt.audio) / 2 / int(utt.info.get("sample_rate", 16000)), 2),
                "reason": "connection_lost",
                "first_audio_latency_ms": None,
                "rms_dbfs": rms_dbfs,
                "peak_dbfs": peak_dbfs,
                "transcript": None,
            })
        print(f"[-] device {peer} disconnected")


async def main():
    try:
        from websockets.asyncio.server import serve  # websockets >= 13
    except ImportError:
        serve = websockets.serve
    async with serve(handler, "0.0.0.0", ARGS.port, max_size=None, ping_interval=10):
        print(f"Listening on port {ARGS.port}. Put one of these into menuconfig -> 'WebSocket server':")
        for ip in local_ips() or ["<your-PC-IP>"]:
            print(f"    ws://{ip}:{ARGS.port}")
        print(f"Recordings are saved in {os.path.abspath(RECORDINGS)}  (Ctrl+C to stop)")
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--asr", action="store_true", help="transcribe with faster-whisper and send the text back")
    ARGS = ap.parse_args()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
