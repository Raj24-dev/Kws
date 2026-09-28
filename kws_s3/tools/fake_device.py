#!/usr/bin/env python3
"""Plays a recording to the cloud server exactly like the ESP32 streams it (tests the server without the board).

    python fake_device.py recording.wav [more.wav ...] [--url ws://127.0.0.1:3000/ws]

Protocol as in kws_s3/main/streamer.c: start message, the first 1 s (pre-roll: the recordings made before
2026-09-28 15:10 start with the wake word) as a burst of 60 ms mu-law messages, the rest in real time as 20 ms
messages, then "end". The server transcribes what follows the pre-roll. Prints the transcript and how long after
the end it came. The WAV must be 16 kHz 16-bit mono (the server's recordings/*.wav are).
The server saves what it receives like any utterance: start it with RECORDINGS_DIR=<some test folder> so test runs
stay out of the real recordings (they feed tools/export_training_clips.py).
"""
import argparse
import asyncio
import json
import time
import wave

import websockets

PREROLL_MS = 1000


def mulaw_encode(x: int) -> int:  # same as mulaw_encode() in kws_s3/main/audio_input.h
    sign = 0x80 if x < 0 else 0
    if sign:
        x = -x
    x = min(x, 32635) + 0x84
    exp = x.bit_length() - 1 - 7
    return ~(sign | (exp << 4) | ((x >> (exp + 3)) & 0x0F)) & 0xFF


async def play(url: str, path: str) -> dict:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1, "need 16 kHz 16-bit mono"
        pcm = w.readframes(w.getnframes())
    ulaw = bytes(mulaw_encode(int.from_bytes(pcm[i:i + 2], "little", signed=True)) for i in range(0, len(pcm), 2))
    out = {"file": path, "text": None}
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "start", "device": "fakedevice01", "wake_word": "marvin", "source": "wake_word",
                                  "score": 0.9, "sample_rate": 16000, "format": "mulaw", "channels": 1,
                                  "preroll_ms": PREROLL_MS, "chunk_ms": 20}))

        async def receive():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "response":
                    out["text"], out["after_end_ms"] = m.get("text"), round((time.perf_counter() - t_end) * 1000)
                    return

        rx = asyncio.create_task(receive())
        pos = 0
        preroll = PREROLL_MS * 16
        while pos < preroll and pos < len(ulaw):  # pre-roll burst
            await ws.send(ulaw[pos:pos + 960])
            pos += 960
        t_live = time.perf_counter()
        while pos < len(ulaw):  # live part in real time
            await ws.send(ulaw[pos:pos + 320])
            pos += 320
            await asyncio.sleep(max(0.0, t_live + (pos - preroll) / 16000 - time.perf_counter()))
        t_end = time.perf_counter()
        await ws.send(json.dumps({"type": "end", "reason": "silence", "duration_ms": pos // 16, "first_audio_latency_ms": 0}))
        await asyncio.wait_for(rx, 30)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--url", default="ws://127.0.0.1:3000/ws")
    args = ap.parse_args()
    for path in args.wav:
        r = asyncio.run(play(args.url, path))
        print(f"{r['file']}: {r['text']!r} ({r.get('after_end_ms')} ms after the end)")


if __name__ == "__main__":
    main()
