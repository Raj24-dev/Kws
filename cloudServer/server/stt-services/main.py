"""Speech-to-text service for the cloud server (faster-whisper, runs on the CPU).

    pip install -r requirements.txt
    uvicorn main:app --port 8000          # the first start downloads Whisper small.en (~0.5 GB)

The wake word is detected by the model on the device, not here. The device streams what is said after it, and this
service writes it down: "Hey Marvin, close the door" -> "Close the door." (if the device also sends a pre-roll, the
transcript still starts at the detection point).

One model, WHISPER_MODEL (small.en): more accurate than base.en on the device's recordings; ~2 s per call on this PC
(Whisper always encodes 30 s of audio). base.en live captions were removed: on this 8-thread CPU they slowed the
small.en calls to 7-8 s after the end of speech.

Two ways in:
  WS   /stream      the live utterance from server.js, frame by frame as the device sends it (16 kHz 16-bit PCM,
                    binary), with JSON control messages:
                      {"type":"start","preroll_ms":0}
                      {"type":"end"}  -> {"type":"final","text":...}           (the command after the wake word)
  POST /transcribe  a WAV file -> {"text":...}   (tools/export_training_clips.py; the whole file, no prompt)
"""
import json
import os
import tempfile
import time

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from starlette.concurrency import run_in_threadpool

MODEL = os.environ.get("WHISPER_MODEL", "small.en")
THREADS = int(os.environ.get("WHISPER_THREADS", str(os.cpu_count() or 4)))  # one call at a time gets every core
BYTES_PER_MS = 32  # 16 kHz, 16-bit
# Whisper copies the style of the text before the audio: without it, long commands came back with no punctuation.
# Neutral words on purpose (it biases the content a little). Measured on 6 device commands: all punctuated, no errors.
PUNCTUATION_PROMPT = "So, what do you think? Well, I'm not sure. Let's see. Okay, go ahead."
QUESTION_WORDS = {"am", "are", "is", "was", "were", "can", "could", "will", "would", "shall", "should", "may", "might",
                  "do", "does", "did", "have", "has", "had", "what", "why", "how", "where", "when", "who", "whom",
                  "whose", "which"}

app = FastAPI()
model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=THREADS)


def transcribe(audio, prompt=None) -> str:
    # temperature 0 = deterministic: the default falls back to random sampling on unsure audio, so the same
    # recording could come back as "Marvin" or "Arvin". Beam 5 is more accurate and, without the fallback, no slower.
    segments, _ = model.transcribe(audio, language="en", beam_size=5, temperature=0.0,
                                   condition_on_previous_text=False, initial_prompt=prompt)
    return " ".join(segment.text.strip() for segment in segments).strip()


def to_float(pcm: bytes):
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def tidy(text: str) -> str:
    """Capital first letter and an end mark if Whisper left it out ("what is your name" -> "What is your name?")."""
    text = text.strip().rstrip(",;:-").strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".?!":
        text += "?" if text.split()[0].lower() in QUESTION_WORDS else "."
    return text


def transcribe_command(pcm: bytes) -> str:
    """The command after the wake word. No speech in it (only "Marvin" was said) -> "": Whisper would invent text
    for silence, with the prompt it even repeats the prompt (seen on 6 of 6 such recordings). The speech check only
    gates: trimming the audio to it made Whisper mishear 2 of 6 commands."""
    audio = to_float(pcm)
    if not len(audio) or not get_speech_timestamps(audio, VadOptions()):
        return ""
    return tidy(transcribe(audio, PUNCTUATION_PROMPT))


assert tidy("what is your name") == "What is your name?" and tidy("close the door,") == "Close the door."
assert tidy("Can you hear me?") == "Can you hear me?" and tidy("") == ""


@app.get("/")
async def health():
    return {"status": "stt-running", "model": MODEL}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    pcm = bytearray()
    command_from = 0  # where the command starts: the detection point (after the pre-roll, if the device sends one)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                pcm += msg["bytes"]
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "start":
                    command_from = int(data.get("preroll_ms") or 0) * BYTES_PER_MS
                elif data.get("type") == "end":
                    t0 = time.perf_counter()
                    text = await run_in_threadpool(transcribe_command, bytes(pcm[command_from:]))
                    await ws.send_json({"type": "final", "text": text, "seconds": round(time.perf_counter() - t0, 2)})
                    await ws.close()  # a clean close: an abrupt one can drop the last message on the client
                    return
    except WebSocketDisconnect:
        return


# Plain "def": FastAPI runs it in a worker thread, so a slow transcription does not block the server
@app.post("/transcribe")
def transcribe_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(file.file.read())
        temp_path = temp.name
    try:
        t0 = time.perf_counter()
        text = transcribe(temp_path)
    finally:
        os.remove(temp_path)
    return {"text": text, "seconds": round(time.perf_counter() - t0, 2)}
