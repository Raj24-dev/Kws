# Benchmarks (SIH PS 26172 audit)

Re-runnable measurements for the kws_s3 firmware + cloudServer. Every number in `../Final Report.md` comes from a
file in `results/` produced by one of these scripts. Evidence labels:

* **[DEVICE-ACOUSTIC]** real sound through the board's microphones (PC speakers or a person)
* **[DEVICE-INJECTED]** test audio fed over the serial port into the board's own pipeline (injection firmware)
* **[HOST-ONLY]** the PC mirror of the firmware pipeline (`kws_s3/tools/check_model.py`)

## Setup (Windows, this PC)

* Board on `COM6` (CH343). ESP-IDF 5.5.5: `. 'C:\Espressif\tools\Microsoft.v5.5.5.PowerShell_profile.ps1'`
* Python for the benchmarks: `C:\Python314\python.exe` with `numpy soundfile pymicro-features ai-edge-litert
  websockets faster-whisper pyserial sounddevice pywin32`
* Server: `cloudServer/server` (`node server.js`, port 3000) and `stt-services` (`python -m uvicorn main:app --port 8000`)
* Wi-Fi: the board joins the laptop's Mobile Hotspot (**2.4 GHz band**), server URI `ws://192.168.137.1:3000/ws`
* Speakers: the laptop's own (WASAPI). The DAC time of every played clip is taken from the audio driver.

## Scripts

| script | measures | evidence |
|---|---|---|
| `run_idle.py --label L --cond quiet\|speech --minutes 6` | idle-listening CPU per core (FreeRTOS idle-task share) and RAM, Wi-Fi + server connected | DEVICE-ACOUSTIC |
| `run_acoustic.py --label L <set.csv\|folder>` | plays clips; detections, latency keyword end -> first audio byte at the server (one PC clock), transcripts, RAM while streaming | DEVICE-ACOUSTIC |
| `run_injected.py --label L --cutoff C <sets>` | TPR / false accepts of the on-device pipeline, and device == mirror check | DEVICE-INJECTED |
| `run_host.py sweep\|test\|faph` | threshold sweep on validation, frozen test, false accepts per hour | HOST-ONLY |
| `analyze_preroll.py val` | detection delay after the keyword end; what streaming from the detection point loses | HOST-ONLY |
| `eval_stt.py`, `eval_stt_filter.py` | server transcripts vs Whisper confidence (hallucination filter check) | HOST-ONLY |
| `make_sets.py` | builds the frozen real validation/test sets (`sets/real_*.csv`, SHA-256 per file) | - |
| `make_tts.py`, `make_latency_clips.py` | SYNTHETIC audio (Windows voices), always labelled synthetic | - |
| `set_config.py` | edits `kws_s3/sdkconfig` options in place (never touches the Wi-Fi credentials) | - |
| `run_final.ps1 -Label L -Level -22.7 -SpeechGain 0.58` | the full re-verification: idle CPU quiet + speech, synthetic latency x2, real test replay | DEVICE-ACOUSTIC |
| `run_trigger.ps1 -Label L -Level -16.7` | streaming path only: 18 loud detections -> board detection -> first send, audio lost, RAM peak | DEVICE-ACOUSTIC |
| `calibrate_level.py --gain 0.5` | received speech level at the board; keep it equal between runs you compare (the level/gain values above are for the board's current position) | - |

## Firmware variants (same sources, `kws_s3/sdkconfig` options)

* production (= `kws_s3/sdkconfig.defaults`): `KWS_PROFILE_OPS=n KWS_INJECT_TEST=n KWS_TELEMETRY_MS=0 KWS_PREROLL_MS=0
  KWS_MIC_SELECT=1 KWS_RAM_LIMIT_KB=256`. The status line's `peak + IRAM code N KB` is the strict RAM reading
  (`ram_strict_peak_kb` in the results).
* profiling: `python set_config.py KWS_PROFILE_OPS=y` -> `[prof]` lines (per-op model time, per-stage feature time)
* injection: `python set_config.py KWS_INJECT_TEST=y` -> no mics/Wi-Fi, audio from the serial port (921600 baud)

Build/flash: `idf.py reconfigure; ninja -C build -j 3; idf.py -p COM6 -b 921600 flash` (restore production options after
a variant).

## Rules followed

* Threshold chosen on validation data only (`run_host.py sweep`: highest cutoff with real-validation TPR >= 95 %);
  the test set (`sets/real_test.csv`) is frozen and reported once per model/cutoff.
* Synthetic audio is used for extra test conditions only and is labelled; no synthetic audio went into training here.
* The real sets are recordings the board itself captured (one speaker, one room) and exist because an earlier model
  fired on them (selection bias toward "hard/known" utterances). They cannot support a false-accepts-per-hour claim.
