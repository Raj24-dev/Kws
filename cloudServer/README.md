# Cloud server for the SIH wake word device

The wake word ("Marvin") is detected by the model on the ESP32; the server does not check it again. After a detection
the ESP32 streams what is said next, the server hands every frame to the speech recogniser the moment it arrives, and
when the speaker stops it writes the command down ("Hey Marvin, close the door" -> "Close the door."), saves it and
sends it back to the board and to the live page.

```
ESP32 (kws_s3) --WebSocket ws://<host>:3000/ws--> server.js --WebSocket (every frame, live)--> stt-services/main.py
     ^                                               |                                          (faster-whisper small.en)
     +------------ {"type":"response"} --------------+--> recordings/<time>_<device>_<wake word>.wav + index.jsonl
```

## Run (two terminals)

```
cd server/stt-services
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000     # first start downloads Whisper small.en (~0.5 GB)
```
```
cd server
npm install
npm start                                          # port 3000
```
Firmware setting (`kws_s3/sdkconfig` or menuconfig): `CONFIG_KWS_SERVER_URI="ws://<this PC's IP>:3000/ws"`.
Without the STT service the server still saves the audio, just without transcripts.

Environment (optional, or a `server/.env` file): `PORT` (3000), `RECORDINGS_DIR` (`recordings`),
`STT_URL` (`http://127.0.0.1:8000`; the server uses its `/stream` WebSocket). For main.py: `WHISPER_MODEL`
(`small.en`), `WHISPER_THREADS` (all CPU threads).

How the transcript is made (measured on this PC, an 8-thread i3-1315U, with device recordings):
* small.en only: more accurate than base.en on the device's commands ("Are you listening to me?" where base.en wrote
  "How are you listening to me?"). ~2 s per call: Whisper always encodes 30 s of audio. The transcript is ready
  ~2.3-3 s after the speaker stops (it was 7-8 s while base.en live captions and a small.en wake word check shared the CPU).
* A neutral punctuated prompt makes Whisper punctuate (long commands came back with none), and a missing end mark is
  added ("what is your name" -> "What is your name?").
* Only "Marvin" and nothing after it -> empty transcript at once. A speech check gates Whisper, which would otherwise
  invent text for the silence (it did on 6 of 6 such recordings).
* If the device sends a pre-roll (`CONFIG_KWS_PREROLL_MS` > 0), the transcript still starts at the detection point.

## Live view
`ws://<host>:3000/live` sends every utterance as it happens: `start` (detected on the device: listening) and `final`
(the transcript). `kws_s3/tools/transcription.html` shows it (dashboard "Transcription" button,
http://localhost:8090/transcription; override: `?asr=ws://<host>:3000/live`).
A page that connects first gets the recent utterances; after a server restart the last 100 come back from
`recordings/index.jsonl`, and an open page keeps what it already shows.

## What gets saved

Each line of `recordings/index.jsonl`: `time, file, device, wake_word, source, score, preroll_ms, duration_s, reason,
first_audio_latency_ms, first_audio_rx_ms, resumes, event_peak, event_ms, truncated, rms_dbfs, peak_dbfs, transcript,
stt_final_ms`. `transcript` = what was said after the wake word; `score` / `event_peak` = the device's model.
Recordings made before 2026-09-28 15:10 also have `verdict` / `verified`: the server's former Whisper check of the wake
word, and their WAVs start with 1 s from before the detection (the wake word itself).

* Look at / listen to everything: `python ../../kws_s3/tools/report.py recordings` then open `recordings/report.html`.
* Turn the labelled recordings into training data for the model:
  `python ../../kws_s3/tools/export_training_clips.py recordings` then use the two zips in the training notebook
  (`MY_RECORDINGS_ZIP`, `MY_NEGATIVES_ZIP`). Only recordings with the wake word in them (a pre-roll) can be used: set
  `CONFIG_KWS_PREROLL_MS=1000` on the device while collecting them.

## Protocol

Device: text `{"type":"start","device","wake_word","score","format":"mulaw","preroll_ms",...}`, then binary audio
(16 kHz mono; G.711 mu-law, 1 byte per sample, with `"format":"mulaw"`, else 16-bit PCM LE; 20 ms per live message),
then `{"type":"end","reason","duration_ms","first_audio_latency_ms","event_peak","event_ms"}`.
Reply after the end: `{"type":"response","text","saved"}`.
Test without the board: `python ../kws_s3/tools/fake_device.py <recording.wav>` (start the server with
`RECORDINGS_DIR=<test folder>` so test runs stay out of the recordings). A `start` with `"source":"resume"` is the rest
of an utterance after a reconnect. Other clients can use `register` / `heartbeat` / `trigger` / binary audio / `stop`.
