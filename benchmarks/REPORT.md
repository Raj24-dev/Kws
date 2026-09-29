# Evaluation report: SIH PS 26172 (keyword "Marvin", ESP32-S3 + INMP441)

Audit -> fix -> re-verify of this repository, 28-30 Sep 2026 (baseline = the firmware exactly as it was flashed
before the audit; the reliability pass of 30 Sep is at the end of section 5). Every number below comes from a file in `benchmarks/results/` produced by a
script in `benchmarks/` (see its README). Evidence labels: **[DEVICE-ACOUSTIC]** sound through the board's
microphone, **[DEVICE-INJECTED]** test audio fed into the board's own pipeline over the serial port,
**[HOST-ONLY]** PC. Confidence: **Certain / Likely / Guessing**. Units: KiB = 1024 B (strictest reading).

Test audio disclosure: all acoustic latency/CPU tests play **synthetic** speech (Windows TTS voices) through the
laptop speaker; the "real" sets are recordings the board itself made of **one speaker in one room**. No synthetic audio
was used for training during this audit (the model's own training used Piper TTS clips; see G2).

---

## 1. Verdict

* **Fully built?** Yes. Board-side KWS (60.9 KB int8 streaming CNN, TFLite Micro) -> WebSocket stream -> Whisper
  server -> transcript works end to end on the physical ESP32-S3 [DEVICE-ACOUSTIC, Certain].
* **Submission-ready?** Gate-compliant now, **not yet competitive on accuracy.** At baseline two gates failed under
  the strict reading (RAM **343 KiB**, CPU **14.4 %**), which caps the score at 30. After the fixes all seven gates pass
  with on-device evidence (RAM **245 KiB, enforced at 256 KiB**; CPU **9.2 %**, max 9.6 %). The biggest remaining
  weakness is the PS's "near-zero false activations": 9 of 17 recorded real false triggers and 45 % of synthetic
  sound-alikes still fire, and there are no hours of real negative audio to prove anything per hour. That needs a
  retrained model and recordings only you can make (section 8).
* **Before -> after:** score **30 (capped; 54 uncapped) -> 68**, remaining potential **~82** (blocked on your
  recordings + a Colab retrain).
* Margins are thin: RAM 245/256 KiB, CPU 9.2-9.6/10 %. Any feature added later must be re-measured.
* **Two microphones (F11, the default since `e10f1ee`)**, re-measured: strict RAM **251 KiB** while streaming, CPU
  **9.34 %** (max **9.9 %**). Both still pass, with less margin; `KWS_MIC_SELECT=1` gives the one-microphone figures.

---

## 2. Requirement matrix (baseline -> final)

Status: BUILT+VERIFIED / BUILT+UNVERIFIED / PARTIAL / MISSING.

| # | PS requirement (atomic) | Baseline | Final | Evidence | Conf. |
|---|---|---|---|---|---|
| R1 | Ultra-lightweight KWS model | BUILT+VERIFIED: 60,896 B int8 MixedNet, arena 25.6 KB used | same model; arena 25,964 of 26,624 B | `kws_s3/model/marvin.tflite`; boot line `model RAM: tensor arena 25964 of 26624 bytes used` | Certain |
| R2 | Runs locally on a low-power device | BUILT+VERIFIED (ESP32-S3 240 MHz, no PSRAM) | same | boot logs in `results/*.log` | Certain |
| R3 | High true-positive rate for "Marvin" | PARTIAL: 33/33 frozen real test [INJECTED] but one speaker; acoustic replay 21/33 | PARTIAL: 33/33 [INJECTED]; 35/40 synthetic [ACOUSTIC]; replay 15/33 | `F5_cut0.6_injected_c0.6.json`, `F8_latency_synth_summary.json`, `final_acoustic_real_test.json` | Likely (1 speaker) |
| R4 | Near-zero false activations | PARTIAL, **not met**: 10/17 real false triggers fire, 50 % synthetic sound-alikes, 2.57 FA/h synthetic speech [HOST] | PARTIAL, **not met**: 9/17, 45 %, 0.64 FA/h [HOST]; no real negative hours | `baseline_refslice_injected_c0.5.json`, `F5_*`, `*_host_faph_*.json` | Certain (not met) |
| R5 | On detection, stream the following audio to a remote ASR | BUILT+VERIFIED | BUILT+VERIFIED (34/35 detections produced a transcript) | `F8_acoustic_latency_synth_pass*.json` | Certain |
| R6 | "Instantly": keyword end -> ASR receives audio | BUILT+VERIFIED: median 140 ms, p95 322 ms | BUILT+VERIFIED: median 122 ms, p95 323 ms | `posB_baseline_latency_summary.json`, `F8_latency_synth_summary.json` | Certain |
| R7 | "Efficiently", minimal data overhead | PARTIAL: G.711 mu-law 128 kbit/s + 15 % framing, stops after 700 ms silence, no pre-roll | same (pre-roll tried, reverted: F10) | `main/streamer.c`, `F10_*` | Certain |
| R8 | Report model size, RAM/flash footprint | BUILT+UNVERIFIED (README counted data only: "228 KB") | BUILT+VERIFIED; strict reading printed by the firmware | status line `peak + IRAM code N KB` | Certain |
| R9 | Report idle-listening CPU | BUILT+UNVERIFIED (one core, quiet room) | BUILT+VERIFIED (both cores, speech, Wi-Fi) | `results/*_idle_*.json` | Certain |
| R10 | Report latency keyword end -> ASR | MISSING (never measured) | BUILT+VERIFIED | `benchmarks/run_acoustic.py` | Certain |
| R11 | Open source only (G1) | BUILT+VERIFIED | same | section 3 | Certain |
| R12 | TinyML framework (TFLite Micro etc.) | BUILT+VERIFIED: esp-tflite-micro 1.4.1 + ESP-NN | same | `kws_s3/dependencies.lock` | Certain |
| R13 | Custom keyword, no pre-trained generic wake word (G2) | BUILT+VERIFIED (disclosure needed), training notebook **outside** the repo | same, notebook now in `training/` | section 3 | Certain |
| R14 | < 256 KB RAM (G3) | **FAIL** strict: 343 KiB | PASS: 245 KiB, enforced | section 4 | Certain |
| R15 | < 10 % CPU idle listening (G4) | **FAIL** strict: 14.4 % | PASS: 9.2 % (max 9.6 %) | section 4 | Certain |
| R16 | No heavy/uncompressed transformer on the edge (G5) | PASS | PASS | section 3 | Certain |
| R17 | Evaluated on a physical MCU (G6) | PASS | PASS | all DEVICE results | Certain |
| R18 | Works for the given keyword (G7) | PASS | PASS | section 3 | Certain |
| R19 | Robust, deployable architecture | PARTIAL: Wi-Fi/URI compiled in, plain `ws://` without auth, laptop hotspot, no OTA; Wi-Fi/WebSocket reconnect works | same | `main/wifi.c`, `main/streamer.c`; reconnects seen after hotspot restarts | Likely |

---

## 3. Gates (baseline -> final)

| Gate | Baseline | Final | Evidence (weakest label) | Conf. |
|---|---|---|---|---|
| G1 Open source only | PASS | PASS | ESP-IDF 5.5.5, esp-tflite-micro 1.4.1, esp-nn 1.4.1, esp_websocket_client 1.8.0, led_strip 3.0.3 (all Apache-2.0); esp-micro-speech-features (TFLM microfrontend, Apache-2.0); server: Node/fastify (MIT), faster-whisper (MIT), Whisper weights (MIT). Searched sources + linker map: no esp-sr/WakeNet/MultiNet, Picovoice/Porcupine, Sensory, Edge Impulse. Training data: Speech Commands CC-BY-4.0; microWakeWord's negative feature sets include **CC-BY-NC** material (fine for SIH, not for a product: disclose). | Certain |
| G2 Custom keyword, trained by us | PASS (disclose) | PASS (disclose) | Trained **from scratch** with microWakeWord @`4665173c` (no pre-trained weights, no checkpoint with a "marvin" class). Positives: Speech Commands v2 "marvin" (**the dataset contains the keyword: disclose**) + 2,000 synthetic Piper voices; negatives: other Speech Commands words, 16 sound-alike phrases x 400 synthetic voices, 14 real board false triggers. `old_marvin.tflite` = our own earlier run, used only for comparison. Board recordings used in training (27 Sep 04:18-06:00) do not overlap the frozen test set (27 Sep 08:12 onward). Notebook: `training/SIH_marvin_retrain_v2.ipynb`. | Certain |
| G3 < 256 KB RAM, strict: IRAM code + static data + peak heap, Wi-Fi up, streaming | **FAIL**: 95.5 + 247 = **343 KiB** (data alone, 247 KB, passes only on the lenient data-only reading) | **PASS: 54.0 + 191 = 245 KiB** peak over 52 streams in 3 runs; `KWS_RAM_LIMIT_KB=256` reserves everything above 256 KiB at boot and no allocation failed, 0 audio lost | `posB_baseline_*` vs `F8_*`, `trig_F8_*` [DEVICE-ACOUSTIC] | Certain |
| G4 < 10 % CPU idle continuous listening, strict: both cores summed, continuous speech in the room, Wi-Fi + server up, 6 min | **FAIL**: 14.4 % mean, 15.4 % max (quiet room 11.7 %) | **PASS**: 9.2 % mean, 9.6 % max (quiet room 8.5 %) | `base_instr_idle_speech.json` vs `F8_idle_speech.json`, `final_idle_quiet.json` [DEVICE-ACOUSTIC] | Certain |
| G5 No heavy transformer on the edge | PASS | PASS | Edge model: 60.9 KB int8 streaming CNN. Whisper small.en (int8) runs only on the ASR server (section 9). | Certain |
| G6 Runs on the physical ESP32-S3 | PASS | PASS | ESP32-S3 QFN56 rev 0.2; all DEVICE results | Certain |
| G7 Works for "Marvin" | PASS | PASS | 33/33 frozen real test clips [DEVICE-INJECTED]; 35/40 synthetic "Marvin, <command>" through the air [DEVICE-ACOUSTIC]; one speaker only | Likely |

---

## 4. Measured numbers

Baseline = firmware as found (cutoff 0.5, telemetry on, both mics). Final = commit `622cba9` (F8) + docs, flashed now.
Acoustic runs compared at the same board position and the same **received** level (checked with
`calibrate_level.py`; the board was re-plugged on 29 Sep, so final runs play 1.3 dB louder to arrive at the same level).

| Metric | Target | Baseline | Final | Evidence |
|---|---|---|---|---|
| Model file (flash) | small | 60,896 B | 60,896 B | file [HOST-ONLY, Certain] |
| App image | - | 1,244,432 B | 1,135,648 B | `idf.py size` [Certain] |
| Tensor arena | - | 30,000 B (25,580 used) | 26,624 B (25,964 used; fast kernels keep their copy plans there) | boot log [DEVICE] |
| IRAM code (in SRAM) | counts | 95.5 KiB | **54.0 KiB** | linker map / `_iram_end - _iram_start` [Certain] |
| Static data (.data + .bss) | counts | 47.0 KiB | 40.5 KiB | linker map [Certain] |
| Data RAM (static + heap), steady / peak while streaming | - | 229-231 / 247 KB | 173 / **191 KB** | status lines [DEVICE-ACOUSTIC] |
| **Strict RAM (IRAM + static + peak heap)** | < 256 KiB | **343 KiB** | **245 KiB** (enforced limit 256, 27-28 KB spare) | [DEVICE-ACOUSTIC] |
| PSRAM | reported separately | 0 (disabled) | 0 (disabled); combined = internal | sdkconfig [Certain] |
| Flash cache (SRAM used as cache, not counted) | disclose | 96 KiB | 96 KiB | sdkconfig [Certain] |
| CPU, speech in room, both cores summed, Wi-Fi up | < 10 % | **14.39 %** (max 15.4) | **9.18 %** (max 9.6) | `*_idle_speech.json` [DEVICE-ACOUSTIC] |
| CPU per core (speech): core 0 / core 1 | - | 4.76 / 9.63 % | 0.78 / 8.40 % | same |
| CPU, quiet room, both cores | < 10 % | 11.73 % (max 13.2) | 8.51 % (max 9.3; measured on the build before F8) | `*_idle_quiet.json` [DEVICE-ACOUSTIC] |
| Model time per inference / features per 30 ms | - | 1.15 ms / 1.4 ms | 0.88 ms / 1.4 ms | status lines [DEVICE] |
| TPR, frozen real test, at the operating cutoff | high | 33/33 = 100 % (cutoff 0.5) | 33/33 = 100 % (cutoff 0.6); 95 % lower bound 89 % | [DEVICE-INJECTED] |
| Real recorded false triggers that still fire (test half) | ~0 | 10/17 = 59 % | 9/17 = 53 % | [DEVICE-INJECTED] |
| Synthetic sound-alikes that fire | ~0 | 60/120 = 50 % | 54/120 = 45 % | [DEVICE-INJECTED, synthetic] |
| False accepts per hour, read speech (3.1 h) | ~0 | 2.57 /h | 0.64 /h (test voice alone: 0.67 -> 0 /h) | [HOST-ONLY, synthetic] |
| False accepts per hour, real room audio | ~0 | not measured | **not measured** (needs H2) | - |
| Detected, synthetic "Marvin, <command>" through the air | high | 33/40 = 82.5 % | 35/40 = 87.5 % | [DEVICE-ACOUSTIC, synthetic] |
| Detected, real test recordings replayed through the speaker | - | 21/33, false 4/17 (other position/level) | 15/33, false 3/17 (at 0.5 it would be 17/33, 4/17) | [DEVICE-ACOUSTIC, weak: re-recorded audio] |
| **Latency keyword end -> first audio at server** (n) | low | **140 ms** median (95 % CI 74-174), p95 322, n = 33 | **122 ms** median (95 % CI 33-187), p95 323, n = 35 | `*_latency_synth_summary.json` [DEVICE-ACOUSTIC] |
| - detection delay (after the word ends) | - | ~120 ms median | ~100 ms median | latency minus the terms below |
| - board detection -> first send (buffering + encoding) | - | 19 ms (p95 20) | 20 ms (p95 21) | board log |
| - network one-way | - | 0.5-2.5 ms (ping RTT median 1-5 ms) | ~1 ms (RTT median 2 ms) | ping during runs |
| Latency error bars | state | playback: WASAPI DAC clock (~1 ms); speaker -> mic < 1 ms; keyword end annotation +-10 ms (10 ms energy frames); server event stamp ~1 ms; all on one PC clock. Trial-to-trial spread (+-100 ms) dominates: see the CIs. | same | method |
| Codec / bitrate | minimal | mu-law 8 bit 16 kHz: 128 kbit/s payload, 20 ms frames = 320 B + 8 B WebSocket + 40 B TCP/IP = 147 kbit/s | same | `main/streamer.c` [Certain] |
| Stream stops at end of speech | yes | yes: 700 ms silence (min 1.5 s, max 8 s) | yes | board log `stream finished (silence)` |
| Audio lost (I2S / network) | 0 | 0 / 0 | 0 / 0 in all final runs | status line [DEVICE] |
| Pre-roll (audio right after the keyword) | no loss | none: the ~100 ms between keyword end and detection is not sent | same (250 ms pre-roll verified on the board but reverted: F10) | [DEVICE-ACOUSTIC] |
| End-to-end transcripts (synthetic commands) | works | 32/33 transcribed, WER 6.5 % | 34/35 transcribed, WER 5.3 % | [DEVICE-ACOUSTIC] |
| Whisper hallucinations in transcripts | 0 | 15 of 79 server transcripts were loops/prompt echoes | filter removes all 15, changes no real command | [HOST-ONLY] |
| Transcript returned after end of speech | - | ~2.3-2.6 s | same | server log |

---

## 5. Fix log (branch `sih-audit-fixes`, one commit per fix)

Protocol for CPU: FreeRTOS idle-task share, 6 min after a 60 s warm-up, synthetic speech looped through the laptop
speaker (the model never pauses), Wi-Fi + server up unless marked "no Wi-Fi". Latency: 20 synthetic
"Marvin, <command>" clips x 2 passes. Accuracy: frozen real val/test sets injected into the board.

| # | What / why | Expected | Before -> after (same protocol) | Kept? | Commit |
|---|---|---|---|---|---|
| T | Measurement harness, frozen real val/test sets (SHA-256), firmware instrumentation (per-task CPU, boot heap marks, injection firmware, op profiler) | - | - | yes | `aeffe49` |
| F1 | Dashboard telemetry (a formatted line + heap/task walks every 100 ms on core 0) off by default | -4 % core 0 | core 0 4.76 -> 0.94 %, sum 14.39 -> 10.09 % (speech); quiet core 0 5.19 -> 0.75 % [DEVICE-ACOUSTIC] | yes | `4aa7c12` |
| F4 | Exact fast kernels for STRIDED_SLICE / CONCATENATION / SPLIT_V (the streaming model's state copies; TFLM recomputed the geometry on every call): `main/fast_ops.cc` | -0.2 ms / inference | model 1.07 -> 0.88 ms; sum 10.09 -> 9.40 % (speech, no Wi-Fi); 226/226 outputs identical to the reference kernels [DEVICE-INJECTED] | yes | `030cbfa` |
| F5 | Cutoff 0.5 -> 0.6, picked on **validation only** (rule fixed beforehand: lowest false accepts among window x cutoff with real-val TPR >= 95 % on the PC mirror) | fewer false activations | val TPR 97.0 -> 93.9 %, val false triggers 76.5 -> 70.6 %; frozen test (once): TPR 100 -> 100 %, false triggers 58.8 -> 52.9 %; sound-alikes 50 -> 45 % [DEVICE-INJECTED]; read speech 2.57 -> 0.64 FA/h [HOST-ONLY]. Latency not worse (final 131.6 / F8 122 vs 140 ms). | yes | `54f0f0d` |
| F7 | Server: drop Whisper hallucinations (looping text, prompt echo) from command transcripts | clean transcripts | 15/15 hallucinated segments removed, 0 real commands changed [HOST-ONLY] | yes | `ee80e50` |
| F2 | IRAM -> flash for Wi-Fi/heap/ringbuf/RMT/event code that never runs with the cache off | -30 KiB strict | IRAM 95.5 -> 65.0 KiB (map, Certain); CPU 9.40 -> 9.33 % (no Wi-Fi) | yes | `6d9a35f` |
| F3 | RAM trims: Wi-Fi static RX 10 -> 6, dynamic RX 32 -> 16, mgmt buffers, A-MPDU RX off, SoftAP/OWE/Enterprise/IPv6/DHCP server off, 4 sockets, NVS off, stacks from measured high-water marks, I2S DMA 6 -> 3 x 20 ms, arena 30 -> 26 KB | -50 KB | Attempt 1 also shrank the TX side (8 dynamic TX, A-MPDU TX off, 2880 B TCP buffers): data peak 247 -> 181 KB **but** first audio 19 -> 144-424 ms after the detection, audio lost (562-1692 frames), WER 6 -> 57 % (`posB_F3c_*`, `trig_expB_*`) -> TX side reverted. Attempt 2: first send 19-20 ms, 0 lost, data peak 199 KB (`trig_expA_*`) | attempt 2 | `6d9a35f` |
| F6 | Server can take a pre-roll (drops the wake word by Whisper word timestamps); board default unchanged | fewer clipped first words | 250 ms pre-roll: command WER 31.7 -> 24.4 % on validation recordings [HOST-ONLY] | server yes | `1d0176c` |
| D1 | README rewritten to match the code and measurements | - | - | yes | `d22d07e` |
| F9 | Training notebook into the repo, with provenance and its known flaw (cutoff from the test split) | reproducibility | - | yes | `b0e8951` |
| V1 | Full re-verification of F1-F7 (`run_final.ps1`) | - | CPU 9.20 % (max 9.5), latency 131.6 ms (p95 299), WER 1.7 %, data peak 199 KB | - | `d4cf2c6` |
| F8 | **Strict RAM under 256 KiB**: one microphone read as I2S mono (-5 KB; averaging both detected no more: 30/40 vs 32/40), FreeRTOS task functions in flash (IRAM 65.0 -> 54.0 KiB), Wi-Fi static RX 6 -> 4 (-3.5 KB); `KWS_RAM_LIMIT_KB` now counts IRAM code and is on by default (256), status line prints the strict peak | pass G3 | strict 264 -> **245 KiB**, enforced; CPU 9.20 -> 9.18 % (max 9.5 -> 9.6); first send 20 ms, 0 lost; detected 32 -> 35 of 40; latency 131.6 -> 122 ms; WER 1.7 -> 5.3 % (one empty transcript + the recurring "set"->"send", both also in the baseline) [DEVICE-ACOUSTIC] | yes | `622cba9` |
| F10 | 250 ms pre-roll on the board | no lost audio after the keyword | first send 20 -> 6 ms; keyword end inside the stream in 29/36 (without: ~100 ms lost in 25/35); strict RAM unchanged (244 KiB); **but** 3/36 transcripts kept a wake-word fragment ("And increase...", "Carmen, send..."), WER 5.3 -> 6.7 %; the benefit (no-pause commands) is only host-verified | **reverted** (rule: no regressions) | `4ba5ff1` |
| D2 | README / Kconfig / benchmarks README match F8 | - | - | yes | `20779ec` |
| F11 | **Two microphones, time-aligned** (hardware: 2 x INMP441, 55 mm apart, shared SCK/WS/SD): the delay between them is measured on speech (cross-correlation of sample differences, cosine-fitted peak; < 0.1 sample in the PC test `kws_s3/test/test_mic_align.c`) and the earlier one is delayed by it (4-tap fractional delay); each microphone is mixed by its own background noise (inverse noise power); the boot rule "within 10 dB, else the louder one" is gone (it picked a faulty, noisy microphone); the audio loop is branch-free and unrolled | no comb filter for off-axis talkers (plain average: notch at 3.1 kHz); a faulty microphone cannot drown the good one; stay under G3/G4 | CPU speech 9.86 % (max 10.6, first version) -> **9.34 % (max 9.9)**, audio stage 0.61 % of a core with one microphone, 0.85 % with two; strict RAM **251 KiB** while streaming (8 streams), first send 19-20 ms, 0 lost [DEVICE-ACOUSTIC] (`F11*_idle_speech`, `F11_micalign_stream_ram.log`). This board's right microphone is 22-31 dB noisier than the left (electrical fault), so it is faded to 0-1 % and the detection A/B (`run_micab.ps1`) waits for a working pair | yes | - |
| X | Adaptive beamforming / noise rejection | - | not attempted (F11 does delay-and-sum only): a 55 mm pair helps little below ~1 kHz, and CPU has ~0.1 % left in the worst 10 s window | - | - |

**Reliability pass, 30 Sep** (same CPU protocol; P0 = two-microphone baseline measured the same day, the model never
paused: 9.57 % mean, highest 10 s window 10.0 %, i.e. the F11 figure of 9.34 % had the model paused 8 % of the time).

| # | What / why | Before -> after | Kept? |
|---|---|---|---|
| R1 | Server: a resumed utterance keeps its live speech-to-text link (before: transcript null and a leaked STT socket); utterances without a live transcript (server restart, broken STT link) are transcribed from the saved file; a disk error or a malformed STT reply no longer stops the server; one transcription at a time | `fake_device.py`: normal, drop + resume, server restart during the gap, two commands at once -> one WAV + correct transcript each [HOST-ONLY] | yes |
| R2 | Audio task only starts the stream and wakes the main task (log + LED moved off the audio path); main loop waits on that notification (100 ms) instead of polling every 20 ms; ping 2 -> 5 s; TCP/IP task pinned to core 0 | core 0 0.86 -> 0.66 %, both cores 9.57 -> 9.41 % (`P0_*`, `P2_*`) [DEVICE-ACOUSTIC] | yes |
| R3 | Task watchdog on the audio task and main loop, panic = restart; restart after 5 s without I2S data; reset reason in the banner; failed heap allocations counted in the status line | fault builds (not committed): a hung audio task restarted the board after 5 s ("task watchdog"), stopped I2S data after 5 s ("software restart") [DEVICE] | yes |
| R4 | Streamer: starts at the detection's ring position; a connection dropped mid-command is resumed on the next connection (reconnect every 200 ms meanwhile) and the chunk that failed is re-sent; `lost_ms` / `resumes` in the end message; streams capped at twice the maximum; red LED when a command could not be delivered | connection cut for 0.5 / 1 / 2 s after the detection (TCP proxy): one recording and one transcript each, lost 588 / 588-648 / 1688 ms = outage minus the 0.47 s ring; before: the rest of the command was lost [DEVICE-ACOUSTIC] | yes |
| R4x | Send timeout 1 -> 4 s (ride out stalls) | every reconnect took 4.0 s: a send that raced the disconnect held the client library's lock for the full timeout | **reverted** |
| R5 | Background-noise estimate starts high instead of at -56 dBFS | right after boot, commands in a -40 dBFS room ran to the 8 s limit instead of ending on silence -> end on silence | yes |
| R6 | Watchdog reset twice a second instead of every 20 ms block | complete firmware: 9.53 -> **9.43 % mean, highest 10 s window 9.9 %** (`P4_*`, `P4b_*`) | yes |
| - | Streaming check of the complete firmware (`trig_P4_*`) | 16 streams, first send 21-26 ms after detection, 0 audio lost, strict RAM peak 253 KiB; 7 failed ~510 B allocations were counted (TCP segments waiting for memory under the 256 KiB limit; the sends were retried) | - |
| - | Latency check (`P4_acoustic_latency_synth_pass1`) | 15/20 detected at -22.7 dB, keyword end -> server median 140 ms (F8: 122 ms, 95 % interval 33-187) | - |
| Y | Frontend window loop, integer square root, FFT (audit items P1, P3, P4) | not changed: the compiler already makes the window loop branch-free, the square root runs on 40 values only, and a bit-exact faster FFT is days of work | - |
| Q1 | Server wake-word removal for a 250 ms pre-roll: 6 rules on cached Whisper words (`eval_strip_rules.py`, `results/strip_rules_eval.txt`) | no rule wins on both sets: dropping leading words that end before the detection + 0.1 s removes the synthetic voices' "And ..." / "Carmen, ..." fragments (board WER 6.6 -> 4.6 %) but cuts real first words in the owner's recordings (val 2 -> 5 cut, WER 24.4 -> 26.8 %) [HOST-ONLY] | current rule kept |
| - | Pre-roll default | stays 0 (owner's decision: only the words after "Marvin" are streamed). For reference, 250 ms improves the command WER on real validation speech from 31.7 to 24.4 % and keeps the ~100 ms after the keyword (`preroll_eval_val_250ms.json`) | - |

---

## 6. Score (out of 100)

Unverified claims earn at most half marks; any failed gate caps the total at 30.

| Category | Max | Baseline | Final | One-line justification (final) |
|---|---|---|---|---|
| Efficiency | 20 | 8 | 16 | 60.9 KB model, strict RAM 245/256 KiB (enforced) and CPU 9.2/10 % measured on the device; margins are thin. |
| Accuracy | 25 | 10 | 11 | TPR 100 % on the frozen test [INJECTED] and 87.5 % through the air, but false activations are far from "near-zero" (53 % of real false triggers, 45 % of sound-alikes) and only one speaker; FA/h is host-only. |
| Latency | 20 | 16 | 16 | 122 ms median / 323 ms p95 keyword end -> server, fully decomposed, one clock; detection delay dominates. |
| Streaming / data overhead | 10 | 6 | 6 | Persistent WebSocket, 20 ms frames, stops at silence, 0 audio lost; but 8-bit mu-law (128 kbit/s) and no pre-roll. |
| End-to-end completeness | 15 | 10 | 11 | Keyword -> transcript works (34/35, WER 5.3 %), hallucinations filtered; ~100 ms after the keyword is not streamed. |
| Engineering, reproducibility, docs | 10 | 4 | 8 | Re-runnable benchmarks with raw logs, frozen hashed sets, notebook in repo, docs match the device, limit enforced in firmware; plain ws:// without auth. |
| **Total** | 100 | **54 -> capped 30** (G3, G4 failed) | **68** (no gate failed) | |

* **Baseline score: 30** (54 before the cap).
* **Final measured score: 68.**
* **Remaining potential: ~82**, blocked on you: retrain v3 with more sound-alike negatives (+4 to +8), real negative
  hours (+2 to +4) and more speakers (+2 to +3) for Accuracy; pre-roll with a robust wake-word strip (+1 to +2);
  ADPCM (+2).

---

## 7. Remaining weak spots, ranked

### (a) Blockers
| # | Weak spot | Evidence | Fix | Effort | Est. gain |
|---|---|---|---|---|---|
| A1 | **False activations are not near-zero**: 9/17 real recorded false triggers and 45 % of synthetic sound-alikes (Martin, Marvel, Kevin, Melvin, Morgan...) fire at 0.6. They score like "Marvin" (up to 0.9+), so no threshold fixes it. | `F5_cut0.6_injected_c0.6.json` [DEVICE-INJECTED] | Retrain v3 with more and heavier sound-alike negatives + the real false triggers (H3); cutoff on validation; test once | M (you: Colab 3-4 h) | Accuracy +4 to +8 |
| A2 | **No real negative hours**: the only FA/h is synthetic read speech on the PC (0.64/h). | `baseline_valcut_host_faph_0.6.json` [HOST-ONLY] | H2: 8 h of real room audio on the board | M (you) | Accuracy +2 to +4 |
| A3 | **One speaker, one room**: TPR by speaker/distance/noise unknown. | `benchmarks/README.md` | H1: >= 10 speakers x 4 distances x noise | M (you) | Accuracy +2 to +3 |

### (b) High-ROI upgrades
| # | Upgrade | Why | Effort | Est. gain |
|---|---|---|---|---|
| B1 | Pre-roll 250 ms with a sturdier wake-word strip on the server (drop a leading word that ends before the detection point + 50 ms or is within edit distance 3 of "marvin"), then re-run F10 and a no-pause test (H1 recordings) | Board side already verified (F10); the leak was 3/36 transcripts | S | Streaming +1, E2E +1 |
| B2 | IMA-ADPCM 4-bit instead of mu-law | 128 -> 64 kbit/s payload (~83 kbit/s on air), negligible CPU on core 0, ~30-line decoder on the server | S-M | Streaming +2 |
| B3 | Lower the detection delay (window 4 or an "instant score > 0.9" rule), chosen on validation | Latency is detection-bound; window 4 cuts the host median delay 120 -> 90 ms but raises FA/h 1.25 -> 1.87 on validation (`sweep_window_val.json`), so only after A1 | S | Latency +1 to +2 |
| B4 | Protect the gate margins: CPU 0.4 % and RAM 11 KiB of headroom | Re-run `run_final.ps1` after any change; the RAM limit is enforced, CPU is not | S | keeps the uncapped score |

### (c) Nice-to-haves
| # | Item | Effort | Est. gain |
|---|---|---|---|
| C1 | `wss://` + a device token (today anyone on the hotspot can inject audio or read `/live`) | M | Engineering +1 |
| C2 | Wi-Fi provisioning instead of credentials compiled in; OTA | M | Engineering +1 |
| C3 | Opus 16 kbit/s (more CPU on core 0, not the audio core) | L | Streaming +1 |
| C4 | Two-mic beamforming: little gain at 55 mm below ~1 kHz and no CPU budget left. Not recommended. | L | ~0 |

---

## 8. Human tasks (exact protocols)

Run from `kws_s3_firmware/` in the ESP-IDF PowerShell unless noted. During any measurement keep the PC otherwise idle
(no Whisper jobs), the hotspot on 2.4 GHz, and the board where it is now (if you move it: `python
benchmarks/calibrate_level.py --gain 0.5` and adjust `-Level/-SpeechGain` until the mic level matches
`benchmarks/results/F8_idle_speech.json` (-30.2 dBFS)).

| # | Task | Protocol | Afterwards |
|---|---|---|---|
| H1 | **Many voices, distances, noise** (the biggest evidence gap) | Recording build: `python benchmarks/set_config.py KWS_PREROLL_MS=1000`, then `idf.py reconfigure; ninja -C build -j 3; idf.py -p <port> -b 921600 flash`. Server running normally (it saves every stream in `cloudServer/server/recordings/`). **>= 10 speakers** (mixed gender/age/accent), each says "Marvin, <any command>" **10 times at 0.5 m, 1 m, 2 m and 3 m** facing the board, then 10 more at 1 m with a fan or TV on; also 5 presses of the BOOT button per speaker without speaking. Write `recordings/sessions.csv` (`start_time,speaker,distance_m,noise`). Include a few commands said **with no pause** after "Marvin" (for B1). | Move the current `benchmarks/sets/real_*.csv` to `benchmarks/sets/v1/` (keep them), run `python benchmarks/make_sets.py` (labels with Whisper, drops doubtful clips, freezes SHA-256), injection build (`python benchmarks/set_config.py KWS_INJECT_TEST=y`, rebuild, flash), `python benchmarks/run_injected.py --label h1 --cutoff 0.6 sets/real_test.csv`. Restore `KWS_INJECT_TEST=n KWS_PREROLL_MS=0`. |
| H2 | **8 h of real negatives** (needed for any "near-zero false activations" claim) | Production firmware; `python tools/serial_log.py <port> --minutes 480` while the room has 2 h conversation (nobody says "Marvin"), 2 h TV/news, 2 h music, 1 h kitchen/fan/traffic noise, 1 h of 3 people reading this list aloud: Martin, Marvel, Marble, Marvelous, Margin, Marlin, Martian, Carving, Starving, Carbon, Garvin, Harvin, Melvin, Kevin, Morgan, Mark, Pardon. | False activations = `WAKE WORD ... DETECTED` lines in `kws_s3/logs/*.log`; FA/h per condition = count / hours. Each detection also saved a WAV on the server: listen and list them in `benchmarks/results/h2_false_activations.csv`. |
| H3 | **Retrain v3** (the only real fix for A1) | Colab, `training/SIH_marvin_retrain_v2.ipynb`, T4 GPU: `RUN_NAME = "marvin_v3"`, `HARD_NEG_WEIGHT = 5`, `CONFUSABLE_SAMPLES = 800`, add "melvin", "kevin", "morgan", "marvelous" to `CONFUSABLE_PHRASES`, and add the H2 false activations to `device_hard_negatives.zip` (`python kws_s3/tools/export_training_clips.py cloudServer/server/recordings`). **Ignore Step 12's suggested cutoff.** | Put `marvin.tflite`/`.json` into `kws_s3/model/`; `python benchmarks/run_host.py sweep --label v3` (validation only) -> set the cutoff in `marvin.json`; injection build: `python benchmarks/run_injected.py --label v3 --cutoff <c> sets/real_test.csv audio/synthetic_confusable audio/synthetic_marvin` **once**; production build: `benchmarks\run_final.ps1 -Label v3 -Level -22.7 -SpeechGain 0.58`. |
| H4 | Hotspot stability | Settings -> Network -> Mobile hotspot: band **2.4 GHz**, **Power saving off** (it switched itself off whenever the board was off Wi-Fi for 5 min). | - |
| H5 | Dashboard | Release the serial port from `tools/dashboard.py` before flashing: press **Reconnect** on http://localhost:8090. Telemetry is off in production (`KWS_TELEMETRY_MS=0`); build with 100 for live graphs (costs ~4 % of core 0). | - |
| H6 | Before any public demo | Keep the server reachable only on the hotspot, or add a token/TLS (C1). | - |

---

## 9. PS ambiguities and the interpretation used (strictest reading)

| Ambiguity | Readings | Used |
|---|---|---|
| "less than 256KB of RAM" | KB = 1000 or 1024 B; data only, or everything in SRAM incl. IRAM code; flash cache counted or not; PSRAM | **KiB; IRAM code + static data + peak heap with Wi-Fi up and streaming.** PSRAM unused (0 B), so internal = combined. The 96 KiB configured as flash cache is hardware cache, not software RAM: excluded, disclosed. |
| "Raspberry Pi or ESP32" | a Pi has >= 512 MB, so the limit only bites on an MCU | Judged on the ESP32-S3; the limit covers the whole edge application, not just the model. |
| "under 10% CPU ... idling in continuous listening" | per core, average of cores, or sum; "idle" = silent room or non-keyword sound | **Both cores summed (% of one core), continuous speech in the room (model never paused), Wi-Fi + server up, >= 6 min, and the highest 10 s window must also stay < 10 %.** Quiet-room numbers reported too. |
| "near-zero false activations" | no number | Needs FA/h on hours of real negatives (H2). Per-clip rates on recorded false triggers and synthetic sound-alikes are reported but are not per hour. **Unverified; per-clip evidence says not met.** |
| "time delta between the keyword ending and the cloud ASR receiving the audio stream" | first byte of any audio, or of post-keyword audio; "ASR" = server socket or recogniser | Keyword end (annotated) -> first audio frame received by the server process, one PC clock. The server hands frames to the recogniser on arrival. |
| "instantly and efficiently stream the subsequent audio" | pre-roll required? | No audio after the detection may be lost (0 lost). The ~100 ms between keyword end and detection is a known gap (F10, B1). |
| "Heavy or uncompressed pre-trained transformers are disqualified" | edge only, or whole system | Edge only (G5 passes). The server's Whisper small.en is a pre-trained transformer used as the **cloud ASR**, which the PS places off-device. A strict judge may still ask: disclosed. |
| "custom key word" | is "Marvin" (a Speech Commands word) custom? | Assigned by SIH, so allowed; trained from scratch; Speech Commands "marvin" clips in training disclosed (G2). |

**Where the earlier framing was wrong** (docs/claims in the repo before this audit):
* "RAM < 256 KB" counted data only (228 KB) and excluded 95.5 KiB of IRAM code "because it is code": strict total
  was **343 KiB**.
* "CPU < 10 %" was one core in a quiet room while core 0 spent ~5 % on dashboard telemetry; strict was **14.4 %**.
* "Two microphones" suggested dual-mic processing; the firmware only averaged them, which measured **no benefit**.
* The training notebook picks the cutoff on its **test** split; `check_model.py` was described as bit-exact with the
  board but differs by up to ~0.03 in score (decisions agreed on all real clips).

---

## Appendix: project map

| Part | Where | Build / run |
|---|---|---|
| Edge firmware (ESP-IDF 5.5.5, C/C++) | `kws_s3/main/` (`main.c` app + status/CPU/RAM accounting, `audio_input.c` I2S + 80 Hz high-pass + mu-law ring, `wake_word.cpp` TFLM + features + 5-window detector + quiet gate, `fast_ops.cc`, `streamer.c` WebSocket, `wifi.c`), `components/esp-micro-speech-features` | ESP-IDF PowerShell: `cd kws_s3; idf.py reconfigure; ninja -C build -j 3; idf.py -p <port> -b 921600 flash` (settings: `sdkconfig.defaults`; Wi-Fi/URI via `idf.py menuconfig`) |
| Model | `kws_s3/model/marvin.tflite` + `.json` (cutoff 0.6) | embedded at build |
| Training | `training/SIH_marvin_retrain_v2.ipynb` (Colab, microWakeWord, int8 streaming export) | Colab T4, 3-4 h |
| Transport | WebSocket `ws://<PC>:3000/ws`, JSON start/stop + binary 20 ms mu-law frames, TCP_NODELAY, ping 5 s, resume after a dropped connection | - |
| ASR server | `cloudServer/server/server.js` (Node/fastify: `/ws` device, `/live` events, saves WAV + `index.jsonl`), `stt-services/main.py` (faster-whisper small.en int8 CPU, VAD gate, hallucination filter) | `node server.js`; `python -m uvicorn main:app --port 8000` |
| Tools | `tools/` (dashboard, serial_log), `training/` (check_model PC mirror, export_training_clips), `cloudServer/tools/` (fake_device, report) | see each folder's README |
| Benchmarks | `benchmarks/` (scripts, frozen sets, `results/` with every log and JSON) | see `benchmarks/README.md` |
