"""Speech-to-text service for the cloud server (faster-whisper, runs on the CPU).

    pip install -r requirements.txt
    uvicorn main:app --port 8000          # the first start downloads Whisper small.en (~0.5 GB)

The wake word is detected by the model on the device, not here. The device streams what is said after it, and this
service writes it down: "Hey Marvin, close the door" -> "Close the door." The device sends a short pre-roll from
before its detection point (so nothing said right after the wake word is lost); the wake word in it is dropped from
the transcript by Whisper's word timestamps.

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
import re
import tempfile
import time

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from starlette.concurrency import run_in_threadpool

MODEL = os.environ.get("WHISPER_MODEL", "small.en")
THREADS = int(os.environ.get("WHISPER_THREADS", str(os.cpu_count() or 4)))  # one call at a time gets every core
WAKE_WORD = os.environ.get("WAKE_WORD", "marvin").lower()  # only used to trim it from the transcript
# Whisper copies the style of the text before the audio: without it, long commands came back with no punctuation.
# Neutral words on purpose (it biases the content a little). Measured on 6 device commands: all punctuated, no errors.
PUNCTUATION_PROMPT = "So, what do you think? Well, I'm not sure. Let's see. Okay, go ahead."
QUESTION_WORDS = {"am", "are", "is", "was", "were", "can", "could", "will", "would", "shall", "should", "may", "might",
                  "do", "does", "did", "have", "has", "had", "what", "why", "how", "where", "when", "who", "whom",
                  "whose", "which"}

app = FastAPI()
model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=THREADS)


def segments_of(audio, prompt=None, words=False):
    # temperature 0 = deterministic: the default falls back to random sampling on unsure audio, so the same
    # recording could come back as "Marvin" or "Arvin". Beam 5 is more accurate and, without the fallback, no slower.
    segments, _ = model.transcribe(audio, language="en", beam_size=5, temperature=0.0,
                                   condition_on_previous_text=False, initial_prompt=prompt, word_timestamps=words)
    return [s for s in segments if is_speech(s)]


def transcribe(audio, prompt=None) -> str:
    return " ".join(s.text.strip() for s in segments_of(audio, prompt)).strip()


PROMPT_WORDS = set(re.findall(r"[a-z']+", PUNCTUATION_PROMPT.lower()))


def is_speech(segment) -> bool:
    """Drops Whisper's two failure modes on the board's short commands, using Whisper's own default thresholds:
    looping text (compression ratio > 2.4: "Okay. Okay. Okay. ...", "Bye. Bye. ...") and the prompt coming back
    (average log-probability < -1.0 AND only prompt words: "Okay, let's see."). Checked on 65 saved utterances:
    removes 10 such texts, keeps every other one (benchmarks/results/stt_filter_check.json)."""
    if segment.compression_ratio > 2.4:
        return False
    words = set(re.findall(r"[a-z']+", segment.text.lower()))
    return not (segment.avg_logprob < -1.0 and words and words <= PROMPT_WORDS)


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


def edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def is_wake_word_part(word: str) -> bool:
    """'Marvin' as Whisper may write it, or its tail when the audio starts inside the word ('vin', 'Arvin')."""
    w = re.sub(r"[^a-z]", "", word.lower())
    return len(w) >= 2 and (edit_distance(w, WAKE_WORD) <= 2 or (WAKE_WORD.endswith(w) and len(w) >= 3))


def transcribe_command(pcm: bytes, detect_s: float = 0.0) -> str:
    """The command after the wake word. detect_s = where the board detected it (= the pre-roll it sent first).
    With a pre-roll the stream also holds the end of the wake word, so nothing said right after it is lost; the
    words that end before the detection point (the wake word itself) are dropped by their Whisper timestamps, and a
    leading piece of the wake word is dropped when the detection fired inside it.
    No speech at all (only "Marvin" was said) -> "": Whisper would invent text for silence, with the prompt it even
    repeats the prompt (seen on 6 of 6 such recordings). The speech check only gates: trimming the audio to it made
    Whisper mishear 2 of 6 commands."""
    audio = to_float(pcm)
    if not len(audio) or not get_speech_timestamps(audio, VadOptions()):
        return ""
    if detect_s <= 0:
        return tidy(transcribe(audio, PUNCTUATION_PROMPT))
    words = [w for s in segments_of(audio, PUNCTUATION_PROMPT, words=True) for w in s.words]
    words = [w for w in words if w.end > detect_s - 0.05]
    while words and words[0].start < detect_s and is_wake_word_part(words[0].word):
        words.pop(0)
    return tidy("".join(w.word for w in words))


assert is_wake_word_part("Marvin,") and is_wake_word_part("vin.") and is_wake_word_part("Marvyn")
assert not is_wake_word_part("in") and not is_wake_word_part("turn") and not is_wake_word_part("Open")
assert tidy("what is your name") == "What is your name?" and tidy("close the door,") == "Close the door."
assert tidy("Can you hear me?") == "Can you hear me?" and tidy("") == ""


@app.get("/")
async def health():
    return {"status": "stt-running", "model": MODEL}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    pcm = bytearray()
    detect_s = 0.0  # where the board detected the wake word, in the stream (= the pre-roll it sends first)
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
                    detect_s = int(data.get("preroll_ms") or 0) / 1000
                elif data.get("type") == "end":
                    t0 = time.perf_counter()
                    text = await run_in_threadpool(transcribe_command, bytes(pcm), detect_s)
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
