# Edge firmware: ESP32-S3 + INMP441

On-device wake word detection (a microWakeWord streaming model running in TensorFlow Lite Micro): the model on the
board decides, and the LED blinks green at every detection. Then what is said after the wake word is streamed over
Wi-Fi (WebSocket) to the speech-recognition server in [`../cloudServer`](../cloudServer), which writes it down
("Marvin, close the door" -> "Close the door.").

```
                 ┌──────────── audio_kws task (core 1) ─────────────┐
INMP441 ─I2S/DMA─► align + mix ─► filter ─► 0.5 s mu-law ring (8 KB)  │
                 │                   └► wake_word: 40 features / 10 ms → int8 → streaming model / 30 ms → average of 5
                 └──────────────────────────────────────────────────┘                                 │ detection
                   streamer (core 0) ◄──────────────────────────────────────────────────────────────┘
                     reads the ring from the detection on ──WebSocket (always open)──► server ──► speech-to-text
                     ◄── {"type":"response","text":"Close the door."} ~2-3 s after the speaker stops
```

| file | what it does |
|---|---|
| `main/main.c` | start-up, audio + detection task, status line, button, RAM limit check |
| `main/audio_input.c` | I2S capture, microphone check at boot, high-pass filter, mu-law ring buffer |
| `main/mic_align.c` | time alignment and noise-weighted mix of the two microphones |
| `main/wake_word.cpp` | feature frontend + TFLite Micro model + sliding-window detector + quiet-room gate |
| `main/fast_ops.cc` | faster, bit-identical kernels for the model's state-copy operations |
| `main/streamer.c` | WebSocket client: streams each command, resumes after a dropped connection |
| `main/wifi.c`, `main/status_led.c`, `main/wav_selftest.c` | Wi-Fi station, RGB LED, boot self-test |
| `components/esp-micro-speech-features` | TensorFlow Lite Micro microfrontend (feature extraction) |
| `model/` | the deployed model (`marvin.tflite` + `marvin.json`) |
| `test/test_mic_align.c` | PC unit test for the microphone alignment |

## 1. Wiring (INMP441 → ESP32-S3 DevKit)

| INMP441 | ESP32-S3 | note |
|---|---|---|
| VDD | 3V3 | **not 5V** |
| GND | GND | |
| L/R | GND (mic 1), 3V3 (mic 2) | selects the left / right I2S slot |
| WS | GPIO 15 | |
| SCK | GPIO 16 | |
| SD | GPIO 17 | |

The pins can be changed in `menuconfig`. Avoid GPIO 0, 3, 45 and 46 (boot pins), 19/20 (USB) and 35–37 (PSRAM on
N8R8/N16R8 modules).

**Two microphones** (default `KWS_MIC_SELECT = 0`) share the same SCK, WS and SD pins: mic 1 has L/R to GND (left
slot), mic 2 has L/R to 3V3 (right slot), and each drives SD only in its own slot. At boot the firmware checks which
slots carry a microphone and uses both, e.g. `microphones: left (L/R to GND) -22.3 dBFS, right (L/R to 3V3) -17.4
dBFS, similarity 0.59 -> using both (time-aligned, mixed by their noise)`. One microphone works too.

**Alignment.** Sharing SCK and WS means both microphones sample at the same instants, so the channels cannot drift.
What differs is when the voice arrives: with the microphones 55 mm apart (`KWS_MIC_SPACING_MM`) a talker off to one
side reaches the nearer one up to 160 µs (2.6 samples) earlier, and a plain average would cancel part of the voice
(a notch at 3.1 kHz for a talker in line with the pair). While someone speaks, the firmware cross-correlates the two
microphones, finds that delay to a fraction of a sample, and delays the earlier microphone by it before mixing
(`main/mic_align.c`), so the voice adds up in phase from any direction.

**Mix.** Each microphone's share follows its background noise (inverse noise power): a healthy pair is mixed about
50/50 and gains ~3 dB SNR; a microphone with a wiring or supply fault is faded out instead of drowning the good one,
and the log says so once. The status line shows the mix, the delay and the talker's angle, e.g. `mics: mix left 54%
right 46% (right noise +0.6 dB), right +115 us after left (talker 46 deg to the left)`.
`KWS_MIC_SPACING_MM = 0` turns the alignment off; `KWS_MIC_SELECT = 1` / `2` reads only that slot (−5 KB RAM).
PC test (no board): `test/test_mic_align.c` (build line in its first comment).

## 2. Model

The deployed model is in `model/` (see [`model/README.md`](model/README.md)). To use another microWakeWord model,
copy its `<name>.tflite` and `<name>.json` there (the training notebook exports both, see [`../training`](../training)).
Optionally add a 16 kHz 16-bit `.wav` of the wake word for the boot self-test. With no model, the firmware runs in
**microphone-test mode**.

## 3. Build, flash, monitor

With the ESP-IDF 5.5 VS Code extension: open this folder, set the target to **esp32s3**, pick the board's port, open
**SDK Configuration Editor → KWS (wake word) settings**, set the Wi-Fi name, password and the server address
`ws://<PC-IP>:3000/ws`, then **Build, Flash and Monitor**. If flashing stalls at `Connecting...`, hold **BOOT**,
tap **RST**, release **BOOT**. Leave the Wi-Fi fields empty to test without Wi-Fi.

From the ESP-IDF terminal:
```
idf.py set-target esp32s3        (first time only; sdkconfig.defaults gives the tested configuration)
idf.py menuconfig                (KWS (wake word) settings)
idf.py -p <port> build flash monitor
```
The Wi-Fi password ends up in `sdkconfig`, which is git-ignored: never commit it. On a PC with 16 GB RAM,
`ninja -C build -j 3` avoids running out of memory while compiling TensorFlow Lite Micro.

## 4. What a healthy run looks like

```
 reset     : power-on                                        <- also "task watchdog", "brownout", "crash (panic)"...
RAM limit test: 163 KB reserved, the firmware has 202 KB (static data + heap) + 54 KB IRAM code = 256 KB to work with
model ready: wake word "marvin", threshold 0.60 (json), window 5, input 3x40 int8
 noise only (3 s)   : PASS - stayed quiet  (highest score 0.01, threshold 0.60)
microphones: left (L/R to GND) -22.3 dBFS, right (L/R to 3V3) -17.4 dBFS, similarity 0.59 -> using both (time-aligned, mixed by their noise)
stream: connected to the server ws://<PC-IP>:3000/ws
[status] up 21s | mic -39.9 dBFS (peak -34.9) | score max 0.00 | detections 0 | inferences 100 (9.9/s; model 0.90 ms
         each, paused in quiet 72%; features 1.30 ms per 30 ms) | CPU core0 0.6% core1 6.4% (...) | RAM used 178 KB
         (peak 181 KB; peak + IRAM code 235 KB of the 256 KB limit, 23 KB free) | wifi OK, server OK | mics: ... |
         audio lost since boot: i2s 0 net 0
>>> WAKE WORD "marvin" DETECTED  (score 0.67, t = 364.76 s)        <- the LED blinks green
stream: first audio sent 22 ms after the detection
score event: peak 1.00 over 870 ms -> detected
stream: stream finished (silence): 2520 ms of audio sent (39 KB), 0 ms lost, 0 resumes
stream: server: {"type":"response","text":"Set an alarm for seven.","saved":"..."}
```
* `audio lost since boot` must stay `i2s 0 net 0`. `failed allocations N` appears only if a heap allocation failed.
* `score event` lines appear for every rise of the score, near-misses too: use them to judge the threshold.
* `[tasks]` (every minute) shows each task's CPU share and free stack; `[heap]` (at boot) the RAM each stage took.

## 5. Design against the SIH limits (CPU < 10 %, RAM < 256 KB, low latency, no lost audio)

Measured on this board with the scripts in [`../benchmarks`](../benchmarks) (report: `../benchmarks/REPORT.md`).
* **One task** reads the microphones and runs the wake word engine (core 1); the I2S DMA buffer (3 × 20 ms) is the
  queue. On a detection it only starts the stream and wakes the main task, which logs and blinks: nothing on the
  audio path waits for the console or the LED. Wi-Fi, the TCP/IP task and the streamer run on core 0.
* **RAM** (no PSRAM), counted strictly = code kept in internal RAM (IRAM, 54 KB) + static data + peak heap (Wi-Fi,
  lwIP, buffers, model, stacks): **253 KiB while streaming**. `KWS_RAM_LIMIT_KB = 256` (on by default) reserves
  everything above 256 KiB at boot, so the firmware cannot use more; the status line shows the strict peak. The
  96 KB configured as flash cache is not counted.
* **CPU** (FreeRTOS idle-task share per core) with continuous speech in the room, Wi-Fi and server connected, two
  microphones: core 0 0.7 %, core 1 8.7 %, **both cores 9.4 % (highest 10 s window 9.9 %)**. Per 30 ms of audio:
  model 0.9 ms + features 1.4 ms. The model pauses after 1.6 s of quiet (sound 6 dB above the background restarts
  it and the paused 300 ms are replayed first). The dashboard telemetry (`KWS_TELEMETRY_MS`) is off by default:
  at 100 ms it costs ~4 % of core 0. The features must stay identical to training (esp-dsp's SIMD FFT changed
  scores by up to 0.26 and was rejected).
* **Latency**: the WebSocket stays open (ping every 5 s), Nagle is off (`TCP_NODELAY`), Wi-Fi power save is off,
  audio goes out in 20 ms messages and the server forwards each frame to the speech recogniser as it arrives. The
  first audio leaves ~25 ms after the detection; from the end of the wake word to the server it is ~120-140 ms
  (median), almost all of it the detector's 5-output average.
* **No lost audio.** The stream starts at the ring position of the detection itself. If the connection drops
  mid-command, the stream waits up to 5 s for it (reconnecting every 200 ms), sends a `"resume"` start and continues
  the same utterance; the server appends it to the same recording. Audio older than the 0.47 s ring is skipped and
  reported per command (`lost_ms`). A command that could not be delivered at all blinks the LED red.
* **Recovery.** The audio task and the main loop are watched by the task watchdog, which restarts the board after
  5 s; 5 s without data from I2S restarts it too. The boot banner prints why the board last restarted.
* **Pre-roll** (`KWS_PREROLL_MS`, default 0): the ~100 ms between the end of "Marvin" and the detection are not
  streamed, so a command said with no pause can lose its first syllable; see the option's help.
* **False activations**: cutoff 0.6 (chosen on validation data). Sound-alike words (Martin, Marvel, Melvin...) still
  fire; the fix is a retrained model with those words as negatives ([`../training`](../training)).
* End of speech = 700 ms of silence (`KWS_STREAM_SILENCE_MS`); a stream lasts 1.5-8 s. A wake word during a stream
  extends it (at most to twice the limit).

## 6. Related tools

* [`../tools`](../tools): live dashboard (needs `KWS_TELEMETRY_MS` > 0) and serial logger.
* [`../training/check_model.py`](../training): runs a model on WAV files with this firmware's pipeline on a PC.
* [`../cloudServer/tools/fake_device.py`](../cloudServer): streams a recording to the server like the board does.
* [`../benchmarks`](../benchmarks): `run_injected.py` feeds test audio into the board's own pipeline over the serial
  port (build with `KWS_INJECT_TEST`); `KWS_PROFILE_OPS` prints per-operation timings. Both are off by default.

## Credits / licences

Feature extraction: `components/esp-micro-speech-features` (TensorFlow Lite Micro microfrontend, Apache-2.0,
fork by Kevin Ahrendt). Model runtime: `espressif/esp-tflite-micro` (Apache-2.0). Detection logic follows ESPHome's
`micro_wake_word` (open source). No proprietary wake word SDK is used.
