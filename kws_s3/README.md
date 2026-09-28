# SIH KWS firmware: ESP32-S3 + INMP441

On-device wake word detection (a microWakeWord streaming model running in TensorFlow Lite Micro): the model on the
board decides, and the LED blinks green at every detection. Then what is said after the wake word is streamed over
Wi-Fi (WebSocket) to a speech-recognition server, which writes it down ("Hey Marvin, close the door" -> "Close the door.").

```
                 ┌──────────── audio_kws task (core 1) ─────────────┐
INMP441 ─I2S/DMA─► filter ─► 0.5 s mu-law ring (8 KB)                │
                 │        └► wake_word: 40 features / 10 ms → int8 → streaming model / 30 ms → average of 5 → threshold
                 └──────────────────────────────────────────────────┘                                 │ detection
                   streamer (core 0) ◄──────────────────────────────────────────────────────────────┘
                     reads the ring from the detection on ──WebSocket (always open)──► cloud server ──► ASR
                     ◄── {"type":"response","text":"Close the door."} ~2-3 s after the speaker stops
```

## 1. Wiring (INMP441 → ESP32-S3 DevKit)

| INMP441 | ESP32-S3 | note |
|---|---|---|
| VDD | 3V3 | **not 5V** |
| GND | GND | |
| L/R | GND | selects the left channel |
| WS | GPIO 15 | |
| SCK | GPIO 16 | |
| SD | GPIO 17 | |

The pins can be changed in `menuconfig`. Avoid GPIO 0, 3, 45 and 46 (boot pins), 19/20 (USB) and 35–37 (PSRAM on N8R8/N16R8 modules).
**Two microphones** (this board: 55 mm apart): wire the second INMP441 to the **same** SCK, WS and SD pins, with its
L/R pin tied to **3V3** (the first one's L/R stays on GND). Each drives SD only in its own slot, so they share the wire.
At boot the firmware checks both slots and prints e.g.
`microphones: left (L/R to GND) -38.2 dBFS, right (L/R to 3V3) -37.9 dBFS, similarity 0.71 -> using both (average)`.
Both are averaged (+3 dB over the microphones' own noise): best for a talker **in front of the pair**, i.e.
perpendicular to the line between the two microphones, with both sound holes facing the same way. If both L/R pins
are on the same level the two microphones fight over SD: the check then shows garbage levels or uses only one.
With a single microphone the firmware simply uses the slot that has it.

## 2. Model

Copy `<name>.tflite` and `<name>.json` from the Colab Step 13 export into `model/`. Optionally, add a 16 kHz
16-bit `.wav` of the wake word there too, for the boot self-test (see `model/README.md`).
With no model, the firmware runs in **microphone-test mode**.

## 3. Build, flash, monitor (VS Code + ESP-IDF extension)

1. Open this folder in VS Code (`File → Open Folder… → kws_s3`).
2. At the bottom bar, set **target = esp32s3**, and pick the **COM port** of the board.
   On a two-port DevKit, either USB-C port works. The port labelled **COM/UART** is the most reliable for flashing.
3. Click **SDK Configuration Editor** (gear icon) → **KWS (wake word) settings**, and set the Wi-Fi name, password and
   server address `ws://<PC-IP>:8765`. Leave them empty to test without Wi-Fi.
   With the cloud server use `ws://<PC-IP>:3000/ws`.
4. Click **Build, Flash and Monitor** (the flame icon).
   If flashing stalls at `Connecting...`, hold **BOOT**, tap **RST**, release **BOOT**.

Terminal alternative (in the ESP-IDF PowerShell from the desktop shortcut):
```
cd kws_s3
idf.py set-target esp32s3        (only the first time; sdkconfig.defaults then gives the audited configuration)
idf.py menuconfig                (KWS (wake word) settings: Wi-Fi name/password, server URI)
idf.py -p COM6 build flash monitor      (use your port; Ctrl+] quits the monitor)
```
On a PC with ~16 GB RAM, `idf.py reconfigure` then `ninja -C build -j 3` avoids running out of memory.

## 4. What a healthy run looks like

```
model ready: wake word "marvin", threshold 0.50 (json), window 5, input 3x40 int8
model RAM: tensor arena 25580 of 30000 bytes used; engine total 41 KB of heap
microphones: left (L/R to GND) -20.5 dBFS, right (L/R to 3V3) -18.3 dBFS, similarity 0.40 -> using both (average)
wifi: connected ... stream: connected to the server ws://<PC-IP>:3000/ws
[status] up 21s | mic -37.5 dBFS (peak -31.1) | score max 0.00 | detections 0 | inferences 248 (24.8/s; model 0.90 ms
         each, paused in quiet 30%; features 1.32 ms per 30 ms) | CPU core0  1.2% core1  7.8% (wake word pipeline 7.54% of
         one core) | RAM used 180 KB (peak 184 KB, 173 KB free) | wifi OK, server OK | audio lost since boot: i2s 0 net 0
[tasks] % of one core (free stack): main@0=0.39(928) ... audio_kws@1=8.19(736) ...   <- every minute: CPU + stack headroom
[heap] Wi-Fi driver + netif           took  36252 B, ...                               <- at boot: RAM per start-up stage
>>> WAKE WORD "marvin" DETECTED  (score 0.64, t = 26.73 s)
stream: first audio sent 19 ms after the detection          <- the LED blinks green at the detection
stream: stream finished (silence): 2500 ms of audio sent
stream: server: {"type":"response","text":"Turn on the lights.","saved":"..."}
```
* `inferences`: the model runs every 30 ms while there is sound, and pauses in quiet (`model paused in quiet`).
  `audio lost since boot` must stay `i2s 0 net 0`.
* `score event` lines appear for every rise of the score, also for near-misses: use them to judge the threshold.

## 5. Audio path: SIH limits (CPU < 10 %, RAM < 256 KB, low latency, no false activations)

All numbers below are measured on this board; the scripts and raw logs are in `../benchmarks/` (see its README and
`../Final Report.md`).
* Gain x1 + 80 Hz high-pass (the INMP441 picks up large infrasonic drift that otherwise clips speech).
* **One task** reads the microphone and runs the wake word engine (core 1); the I2S DMA buffer (3 x 20 ms) is the
  queue. Wi-Fi and the streamer run on core 0.
* **RAM** (no PSRAM): counted strictly = code the chip keeps in internal RAM (IRAM, 65 KB) + static data + peak heap
  (Wi-Fi, lwIP, buffers, model, stacks). Measured with Wi-Fi up and streaming: 181 KB data in use, 199 KB peak, i.e.
  264 KB with IRAM (still above 256 KB counted this way; 199 KB counting data only). The mu-law ring holds 0.5 s
  (8 KB); it grows to 16/32 KB only for pre-rolls above 250/750 ms. `[heap]` lines at boot show what each start-up
  stage takes; `KWS_RAM_LIMIT_KB = 256` (menuconfig) reserves everything above 256 KB of data RAM at boot.
* **CPU**: FreeRTOS idle-task share per core. With continuous speech in the room (model never paused) and Wi-Fi up:
  core 0 0.9 %, core 1 8.4 %, both cores together 9.2-9.5 %. Per 30 ms of audio: model 0.88 ms + features 1.4 ms.
  The model's state-update copies use exact fast kernels (`main/fast_ops.cc`). The dashboard telemetry
  (`KWS_TELEMETRY_MS`) is off by default: at 100 ms it cost ~4 % of core 0. The model pauses after 1.6 s of quiet
  (sound 6 dB above the background restarts it, and the paused 300 ms are replayed first, so the start of a word is
  kept). The features must stay identical to training: esp-dsp's SIMD FFT was tried and rejected (score changes up to
  0.26).
* **Latency**: the WebSocket stays open (ping every 2 s), Nagle is off (`TCP_NODELAY`), audio goes out in 20 ms
  messages, and the server hands every frame to the speech recogniser the moment it arrives. Measured (keyword end ->
  first audio frame at the server, one PC clock): median ~140-150 ms, p95 ~320 ms; the board sends its first audio
  19 ms after the detection. Keep the Wi-Fi TX buffers / TCP buffers at their defaults: smaller ones delayed the
  first audio by 150-400 ms.
* **False activations**: the model on the board is the only judge. Cutoff 0.6 (chosen on validation data): on the
  frozen real test set (board injection) 33/33 "Marvin" detected, but 9 of 17 recorded sound-alike false triggers
  still fire, and 45 % of synthetic sound-alikes (Martin, Marvel, Kevin, Melvin, Morgan...). Fewer false activations
  need a retrained model with those words as negatives.
* End of speech = 700 ms of silence (`KWS_STREAM_SILENCE_MS`). A wake word during a stream extends the stream.

## 6. Tools and server

* `../cloudServer`: the server (Node + Whisper small.en). Transcribes and saves every command. See its README.
* `tools/dashboard.cmd` (double-click) or `tools/dashboard.py [COM6] [--reset] [--host 0.0.0.0]`: finds the board's
  USB port by itself and opens a live web page (http://localhost:8090 — not the .html file) with the wake-word
  score, CPU and RAM (lowest/highest), mic level, board health and a tester scorecard: press Space each time you say
  the wake word -> correct / false alarms / missed, precision, detection rate, false alarms per hour, "Export CSV".
  Saves the same log as `serial_log.py`. Needs `KWS_TELEMETRY_MS` > 0 (e.g. 100 ms: one `@T ...` line per tick;
  default 0 = off, see CPU above). It owns the COM port: press "Release port" before flashing. `--selftest` checks the scoring.
  The "Transcription" button opens http://localhost:8090/transcription in a new page: "listening" as soon as the
  board detects the wake word, then what was said after it (Whisper small.en, punctuated) when the speaker stops,
  above everything the server has saved (`recordings/index.jsonl` + WAV, last 100 reloaded after a restart).
  It connects to `ws://<this PC>:3000/live` by itself (`?asr=ws://<host>:3000/live` to override).
* `tools/serial_log.py COM6 [--reset] [--minutes N]`: serial monitor that also saves timestamped logs in `logs/`.
* `tools/report.py <recordings>`: HTML page with audio player, spectrogram and metadata for every utterance.
* `tools/export_training_clips.py <recordings>`: cuts the wake word out of the Whisper-labelled recordings into
  `device_positives.zip` / `device_hard_negatives.zip` for the training notebook (needs recordings made with a
  pre-roll: `KWS_PREROLL_MS=1000`).
* `tools/ws_server.py`: minimal stand-alone test server (saves WAVs, optional Whisper), protocol-compatible.
* `tools/fake_device.py <wav...>`: plays recordings to the cloud server exactly like the board streams them (mu-law,
  pre-roll burst, real time) and prints the transcript: tests the server without the board. Start the server with `RECORDINGS_DIR=<test folder>` so test runs stay out of the training data.
* `tools/check_model.py`: runs a model on WAV files with the firmware's pipeline on a PC or in Colab. Close to, but
  not bit-identical with, the board (scores differ by up to ~0.03; decisions agreed on all real test clips). For
  on-device numbers use `../benchmarks/run_injected.py` with the injection firmware.

## Credits / licences
Feature extraction: `components/esp-micro-speech-features` (TensorFlow Lite Micro microfrontend, Apache-2.0,
fork by Kevin Ahrendt). Model runtime: `espressif/esp-tflite-micro` (Apache-2.0). Detection logic follows ESPHome's
`micro_wake_word` (open source). No proprietary wake word SDK is used.
