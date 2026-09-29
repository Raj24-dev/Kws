# Server and speech recognition

The wake word ("Marvin") is detected by the model on the ESP32; the server does not check it again. After a detection
the ESP32 streams what is said next, the server hands every frame to the speech recogniser the moment it arrives, and
when the speaker stops it writes the command down ("Marvin, close the door" -> "Close the door."), saves it and sends
it back to the board and to the live page.

```
ESP32 (kws_s3) --WebSocket ws://<host>:3000/ws--> server.js --WebSocket (every frame, live)--> stt-services/main.py
     ^                                               |                                          (faster-whisper small.en)
     +------------ {"type":"response"} --------------+--> recordings/<time>_<device>_<wake word>.wav + index.jsonl
```

| path | what it is |
|---|---|
| `server/server.js` | WebSocket gateway for the boards (Fastify): saves each command, streams it to speech-to-text |
| `server/stt-services/main.py` | speech-to-text service (FastAPI + faster-whisper small.en, CPU) |
| `tools/fake_device.py` | plays a recording to the server exactly like the board streams it, also with a dropped connection |
| `tools/report.py` | HTML page with player, spectrogram and metadata for every saved command |

## Run (two terminals)

```
cd server/stt-services
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000     # the first start downloads Whisper small.en (~0.5 GB)
```
```
cd server
npm install
npm start                                          # port 3000
```
Firmware setting (menuconfig, KWS settings): server address `ws://<this PC's IP>:3000/ws`. Without the STT service the
server still saves the audio, just without transcripts.

Environment (optional, or a `server/.env` file): `PORT` (3000), `RECORDINGS_DIR` (`recordings`), `STT_URL`
(`http://127.0.0.1:8000`; the server uses its `/stream` WebSocket). For `main.py`: `WHISPER_MODEL` (`small.en`),
`WHISPER_THREADS` (all CPU threads), `WAKE_WORD` (`marvin`, only used to trim it from transcripts).

## How the transcript is made

* **small.en** on the CPU (int8). Whisper always encodes 30 s of audio, so a call takes ~2 s on an 8-thread laptop
  CPU and the transcript is ready ~2-3 s after the speaker stops. small.en was more accurate than base.en on the
  board's commands. Transcriptions run one at a time in arrival order: each one uses every CPU thread, so two at
  once would both finish late.
* A neutral punctuated prompt makes Whisper punctuate, and a missing end mark is added ("what is your name" ->
  "What is your name?"). Temperature 0 (deterministic), beam 5.
* A speech check (Silero VAD) gates Whisper: "Marvin" with nothing after it gives an empty transcript at once
  instead of invented text. Looping text and a repeated prompt are filtered out.
* If the board sends a pre-roll (`KWS_PREROLL_MS` > 0), the words before the detection point (the wake word) are
  dropped by their Whisper timestamps.

## Reliability

* **Dropped connections.** When a board's connection breaks mid-command, the recording is kept (as a `.part` file)
  and waits 30 s for the board to resume it; the board then sends `"source":"resume"` and the audio is appended to
  the same recording, which keeps its live speech-to-text link, so the result is one WAV and one transcript. If no
  resume comes, it is saved as it is (`reason: connection_lost`).
* **Server restart.** `.part` files left from before a restart get the same 30 s to be resumed. A recording without
  a live transcript (after a restart, or when the STT link broke) is transcribed from the saved audio at the end.
* A malformed STT reply or a disk error while saving audio is logged and does not stop the server.

Test without the board: `python tools/fake_device.py <recording.wav>`, or with `--drop-at-ms 1000` to cut the
connection after 1 s and resume. Start the server with `RECORDINGS_DIR=<test folder>` so test runs stay out of the
real recordings.

## Live view

`ws://<host>:3000/live` sends every command as it happens: `start` (the board detected the wake word) and `final`
(the transcript). [`../tools/transcription.html`](../tools) shows it. A page that connects first gets the recent
commands; after a server restart the last 100 come back from `recordings/index.jsonl`.

## What gets saved

Each line of `recordings/index.jsonl`: `time, file, device, wake_word, source, score, preroll_ms, duration_s, reason,
first_audio_latency_ms, first_audio_rx_ms, resumes, lost_ms, event_peak, event_ms, truncated, rms_dbfs, peak_dbfs,
transcript, stt_final_ms`. `transcript` = what was said after the wake word; `score` / `event_peak` = the board's
model; `lost_ms` = audio the board could not deliver (network outage longer than its buffer).

* Listen to everything: `python tools/report.py` (default `server/recordings`), then open `server/recordings/report.html`.
* Turn labelled recordings into training data: [`../training/export_training_clips.py`](../training) (needs
  recordings made with a pre-roll that contains the wake word: `KWS_PREROLL_MS=1000` on the board).

## Protocol

Board -> server: text `{"type":"start","device","wake_word","source","score","format":"mulaw","preroll_ms",...}`,
then binary audio (16 kHz mono; G.711 mu-law, 1 byte per sample, with `"format":"mulaw"`, else 16-bit PCM LE; 20 ms
per live message), then `{"type":"end","reason","duration_ms","first_audio_latency_ms","lost_ms","resumes",...}`.
Reply after the end: `{"type":"response","text","saved"}`. A `start` with `"source":"resume"` continues the
interrupted utterance. Other clients can use `register` / `heartbeat` / `trigger` / binary audio / `stop`.
