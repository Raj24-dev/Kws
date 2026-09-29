#!/usr/bin/env python3
"""Plays a recording to the cloud server exactly like the ESP32 streams it (tests the server without the board).

    python fake_device.py recording.wav [more.wav ...] [--url ws://127.0.0.1:3000/ws] [--preroll-ms 0]
    python fake_device.py recording.wav --drop-at-ms 1500     # connection lost mid-command, then resumed

Protocol as in kws_s3/main/streamer.c: start message, the pre-roll (if any) as a burst of 60 ms mu-law messages,
the rest in real time as 20 ms messages, then "end". With --drop-at-ms the connection closes after that much audio,
and a second connection continues the same utterance with "source":"resume": the server must save one WAV and
return one transcript. Prints the transcript, the saved file and how long after the end the transcript came.
The WAV must be 16 kHz 16-bit mono (the server's recordings/*.wav are; the ones made before 2026-09-28 15:10 start
with 1 s of pre-roll, so play those with --preroll-ms 1000).
The server saves what it receives like any utterance: start it with RECORDINGS_DIR=<some test folder> so test runs
stay out of the real recordings (they feed training/export_training_clips.py).
"""
import argparse
import asyncio
import json
import time
import wave

import websockets


def mulaw_encode(x: int) -> int:  # same as mulaw_encode() in kws_s3/main/audio_input.h
    sign = 0x80 if x < 0 else 0
    if sign:
        x = -x
    x = min(x, 32635) + 0x84
    exp = x.bit_length() - 1 - 7
    return ~(sign | (exp << 4) | ((x >> (exp + 3)) & 0x0F)) & 0xFF


async def send_audio(ws, ulaw: bytes, pos: int, end: int, preroll: int) -> int:
    """Sends ulaw[pos:end]: pre-roll as a burst of 60 ms messages, the rest in real time as 20 ms messages."""
    t0, live0 = time.perf_counter(), max(pos, preroll)
    while pos < end:
        n = min(960 if pos < preroll else 320, end - pos)
        await ws.send(ulaw[pos:pos + n])
        pos += n
        if pos > preroll:
            await asyncio.sleep(max(0.0, t0 + (pos - live0) / 16000 - time.perf_counter()))
    return pos


async def play(url: str, path: str, preroll_ms: int, drop_at_ms: int, gap_ms: int) -> dict:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1, "need 16 kHz 16-bit mono"
        pcm = w.readframes(w.getnframes())
    ulaw = bytes(mulaw_encode(int.from_bytes(pcm[i:i + 2], "little", signed=True)) for i in range(0, len(pcm), 2))
    start = {"type": "start", "device": "fakedevice01", "wake_word": "marvin", "source": "wake_word", "score": 0.9,
             "sample_rate": 16000, "format": "mulaw", "channels": 1, "preroll_ms": preroll_ms, "chunk_ms": 20}
    preroll, pos, resumes = preroll_ms * 16, 0, 0
    if 0 < drop_at_ms * 16 < len(ulaw):
        async with websockets.connect(url) as ws:  # the connection breaks mid-command ...
            await ws.send(json.dumps(start))
            pos = await send_audio(ws, ulaw, 0, drop_at_ms * 16, preroll)
        await asyncio.sleep(gap_ms / 1000)
        start, resumes = {**start, "source": "resume"}, 1  # ... and the device resumes the same utterance

    out = {"file": path, "text": None, "saved": None}
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(start))
        t_end = None

        async def receive():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "response":
                    out.update(text=m.get("text"), saved=m.get("saved"),
                               after_end_ms=round((time.perf_counter() - t_end) * 1000))
                    return

        rx = asyncio.create_task(receive())
        pos = await send_audio(ws, ulaw, pos, len(ulaw), preroll)
        t_end = time.perf_counter()
        await ws.send(json.dumps({"type": "end", "reason": "silence", "duration_ms": pos // 16,
                                  "first_audio_latency_ms": 0, "lost_ms": 0, "resumes": resumes}))
        await asyncio.wait_for(rx, 60)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--url", default="ws://127.0.0.1:3000/ws")
    ap.add_argument("--preroll-ms", type=int, default=0, help="audio at the start of the file from before the detection")
    ap.add_argument("--drop-at-ms", type=int, default=0, help="close the connection after this much audio, then resume")
    ap.add_argument("--gap-ms", type=int, default=500, help="time between the drop and the resume")
    args = ap.parse_args()
    for path in args.wav:
        r = asyncio.run(play(args.url, path, args.preroll_ms, args.drop_at_ms, args.gap_ms))
        print(f"{r['file']}: {r['text']!r} -> {r['saved']} ({r.get('after_end_ms')} ms after the end)")


if __name__ == "__main__":
    main()
