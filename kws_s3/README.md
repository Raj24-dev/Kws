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
**Two microphones are used** (default `KWS_MIC_SELECT = 0`): both INMP441 share the **same** SCK, WS and SD pins;
mic 1 has L/R to GND (left slot), mic 2 has L/R to **3V3** (right slot), each drives SD only in its own slot. The
firmware reads both slots, checks at boot that each one carries a microphone, and uses both, e.g. `microphones: left
(L/R to GND) -19.3 dBFS, right (L/R to 3V3) -11.4 dBFS, similarity -0.00 -> using both (time-aligned, mixed by their
noise)`. (The boot levels are mostly the INMP441's power-up drift; they only tell a microphone from an empty slot.) If
both L/R pins are on the same level the two microphones fight over SD: the check then shows garbage levels.

**Synchronisation.** Sharing SCK and WS means both microphones sample at the same instants: every I2S frame holds one
left and one right sample taken together, so the two channels cannot drift or slip against each other. What is not
the same is when the voice arrives: with the microphones **55 mm** apart (`KWS_MIC_SPACING_MM`) a talker off to one
side reaches the nearer one up to 160 us (2.6 samples) earlier, and a plain average then cancels part of the voice
(in line with the pair: a notch at 3.1 kHz). So while someone speaks (20 ms blocks 12 dB above the background, one estimate
per 80 ms of speech) the firmware cross-correlates the two microphones, finds that delay to a fraction of a sample, and delays the earlier
microphone by it (4-tap fractional-delay filter) before mixing (`main/mic_align.c`). The voice then adds up in
phase from any direction. Sounds the two do not hear alike (wind, touching one microphone) do not move the delay. The
mix trails the later microphone by one sample (0.06 ms).

**Mix.** Each microphone's share follows its own background noise (inverse noise power): a healthy pair is mixed
50/50 and gains 3 dB of SNR; a microphone with a wiring or supply fault that adds noise is faded out instead of
drowning the good one (20 dB noisier: 1 %; a fault that starts later is followed within ~25 s), and the log says so
once: `the right microphone is 31 dB noisier than the left one (electrical, not sound), so it gets only 0% of the mix.
Check its VDD/GND/SD wires and that its L/R pin is firmly on 3V3`. The status line shows both, e.g.
`mics: mix left 50% right 50% (right noise +0.4 dB), right +46 us after left (talker 17 deg to the left), 212
estimates`: walk around the board and the angle follows you. PC test (no board): `test/test_mic_align.c`, build line
in its first comment. `KWS_MIC_SPACING_MM = 0` turns the time alignment off, `KWS_MIC_SELECT = 1` / `2` reads only
that slot (I2S mono, -5 KB RAM). The earlier A/B on this board (plain average, before the alignment) detected no
more than the left microphone alone (30/40 vs 32/40 synthetic clips, `../benchmarks/results/micab_*`);
`../benchmarks/run_micab.ps1` now compares aligned, unaligned and left only.

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
RAM limit test: 163 KB reserved, the firmware has 202 KB (static data + heap) + 54 KB IRAM code = 256 KB to work with
model ready: wake word "marvin", threshold 0.60 (json), window 5, input 3x40 int8
model RAM: tensor arena 25964 of 26624 bytes used; engine total 38 KB of heap
microphones: left (L/R to GND) -21.7 dBFS, right (L/R to 3V3) -120.0 dBFS (not used), similarity 0.00 -> using left
wifi: connected ... stream: connected to the server ws://<PC-IP>:3000/ws
[status] up 21s | mic -30.1 dBFS (peak -26.0) | score max 0.00 | detections 0 | inferences 333 (33.3/s; model 0.87 ms
         each, paused in quiet 0%; features 1.39 ms per 30 ms) | CPU core0  0.9% core1  8.4% (wake word pipeline 8.16% of
         one core) | RAM used 173 KB (peak 186 KB; peak + IRAM code 240 KB of the 256 KB limit, 27 KB free) | wifi OK,
         server OK | audio lost since boot: i2s 0 net 0
[tasks] % of one core (free stack): main@0=0.31(928) ... audio_kws@1=8.29(736) ...   <- every minute: CPU + stack headroom
[heap] Wi-Fi driver + netif           took  37892 B, ...                               <- at boot: RAM per start-up stage
>>> WAKE WORD "marvin" DETECTED  (score 0.64, t = 26.73 s)
stream: first audio sent 20 ms after the detection          <- the LED blinks green at the detection
stream: stream finished (silence): 2500 ms of audio sent
stream: server: {"type":"response","text":"Turn on the lights.","saved":"..."}
```
* `inferences`: the model runs every 30 ms while there is sound, and pauses in quiet (`model paused in quiet`).
  `audio lost since boot` must stay `i2s 0 net 0`.
* `score event` lines appear for every rise of the score, also for near-misses: use them to judge the threshold.

## 5. Audio path: SIH limits (CPU < 10 %, RAM < 256 KB, low latency, no false activations)

All numbers below are measured on this board; the scripts and raw logs are in `../benchmarks/` (see its README and
`../Final Report.md`).
* Two microphones time-aligned to the talker and averaged (section 1), then gain x1 + 80 Hz high-pass (the INMP441
  picks up large infrasonic drift that otherwise clips speech).
* **One task** reads the microphone and runs the wake word engine (core 1); the I2S DMA buffer (3 x 20 ms) is the
  queue. Wi-Fi and the streamer run on core 0.
* **RAM** (no PSRAM): counted strictly = code the chip keeps in internal RAM (IRAM, 54 KB) + static data + peak heap
  (Wi-Fi, lwIP, buffers, model, stacks), in KiB. Measured with Wi-Fi up and streaming, two microphones: 197 KB data
  peak, **251 KB with the IRAM code** (one microphone, `KWS_MIC_SELECT = 1`: 191 / 245 KB). `KWS_RAM_LIMIT_KB = 256` (on by default) reserves everything above 256 KB at
  boot, so the firmware cannot use more; the status line shows the strict peak against it. The mu-law ring holds 0.5 s
  (8 KB); it grows to 16/32 KB only for pre-rolls above 250/750 ms. `[heap]` lines at boot show what each start-up
  stage takes. The 96 KB configured as flash cache (32 KB instruction + 64 KB data) is not counted.
* **CPU**: FreeRTOS idle-task share per core. With continuous speech in the room (model never paused) and Wi-Fi up,
  two microphones: core 0 0.8 %, core 1 8.5 %, both cores together **9.3 % (highest 10 s window 9.9 %)** (one
  microphone: 9.2 %, highest 9.6 %). Per 30 ms of audio: model 0.88 ms + features 1.4 ms; the audio stage (both
  microphones, alignment, filter) takes 0.85 % of a core. Its loops have no branches and do two samples per pass:
  the FPU waits for each result, so overlapping two samples is what makes two microphones affordable.
  The model's state-update copies use exact fast kernels (`main/fast_ops.cc`). The dashboard telemetry
  (`KWS_TELEMETRY_MS`) is off by default: at 100 ms it cost ~4 % of core 0. The model pauses after 1.6 s of quiet
  (sound 6 dB above the background restarts it, and the paused 300 ms are replayed first, so the start of a word is
  kept). The features must stay identical to training: esp-dsp's SIMD FFT was tried and rejected (score changes up to
  0.26).
* **Latency**: the WebSocket stays open (ping every 2 s), Nagle is off (`TCP_NODELAY`), audio goes out in 20 ms
  messages, and the server hands every frame to the speech recogniser the moment it arrives. Measured (keyword end ->
  first audio frame at the server, one PC clock, 35 synthetic trials): median 122 ms (95 % interval 33-187 ms), p95
  323 ms. Almost all of it is the detection itself (the 5-output average crosses the threshold ~100-130 ms after the
  word ends); the board sends its first audio 20 ms after the detection and the network adds 1-2 ms. Keep the Wi-Fi
  TX buffers / TCP buffers at their defaults: smaller ones delayed the first audio by 150-400 ms.
* **Pre-roll** (`KWS_PREROLL_MS`, default 0): the audio between the end of "Marvin" and the detection (~100 ms) is not
  streamed, so a command said with no pause can lose its first syllable. 250 ms fixes that on the board, but the
  server's wake-word removal still leaks a fragment in ~1 of 12 transcripts; see the option's help.
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
