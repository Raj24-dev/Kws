# Benchmarks

Re-runnable measurements for the ESP32-S3 firmware (`firmware/`) + cloud server (`server/`). Every number in [`REPORT.md`](REPORT.md) comes from a
file in `results/` produced by one of these scripts. Evidence labels:

* **[DEVICE-ACOUSTIC]** real sound through the board's microphones (PC speakers or a person)
* **[DEVICE-INJECTED]** test audio fed over the serial port into the board's own pipeline (injection firmware)
* **[HOST-ONLY]** the PC mirror of the firmware pipeline (`training/check_model.py`)

Raw serial logs (`*.log`, `*.out`) that the run scripts write next to the JSON summaries are not versioned; the JSON files are the results, and the original raw logs remain in git history.

## Setup (Windows)

* Board on a USB serial port (the scripts default to `COM6`; pass `--port`). ESP-IDF 5.5 environment for building.
* Python 3: `pip install -r benchmarks/requirements.txt` (the PowerShell scripts use the `python` on your PATH, or the one in `$env:PYTHON`; inside the ESP-IDF shell
  `python` is the IDF environment, so set `$env:PYTHON` to the interpreter that has these packages).
* Server: `server/` (`node server.js`, port 3000) and `stt-services` (`python -m uvicorn main:app --port 8000`)
* Wi-Fi: the board joins the laptop's Mobile Hotspot (**2.4 GHz band**), server URI `ws://<PC-IP>:3000/ws`
  (the PC's address on that network). Windows turns the hotspot off after ~5 min without clients, e.g. while the injection
  firmware runs: turn it back on before the next acoustic run.
* Speakers: the laptop's own (WASAPI). The DAC time of every played clip is taken from the audio driver.
* Keep the PC otherwise idle during acoustic runs (a parallel Whisper job starves the STT service and skews timing),
  and compare firmware versions only at the same speaker level and board position (`calibrate_level.py`).

## Scripts

| script | measures | evidence |
|---|---|---|
| `run_idle.py --label L --cond quiet\|speech --minutes 6` | idle-listening CPU per core (FreeRTOS idle-task share) and RAM, Wi-Fi + server connected | DEVICE-ACOUSTIC |
| `run_acoustic.py --label L <set.csv\|folder>` | plays clips; detections, latency keyword end -> first audio byte at the server (one PC clock), transcripts, RAM while streaming | DEVICE-ACOUSTIC |
| `run_injected.py --label L --cutoff C <sets>` | TPR / false accepts of the on-device pipeline, and device == mirror check | DEVICE-INJECTED |
| `run_host.py sweep\|test\|faph` | threshold sweep on validation, frozen test, false accepts per hour | HOST-ONLY |
| `analyze_preroll.py val` | detection delay after the keyword end; what streaming from the detection point loses | HOST-ONLY |
| `eval_preroll.py val 0.25` | command WER with and without a pre-roll, with the server's real transcription code | HOST-ONLY |
| `eval_strip_rules.py` | rules for dropping the wake word from a pre-roll transcript (real validation speech + the board's F10 recordings) | HOST-ONLY |
| `eval_stt.py`, `eval_stt_filter.py` | server transcripts vs Whisper confidence (hallucination filter check) | HOST-ONLY |
| `make_sets.py` | builds the frozen real validation/test sets (`sets/real_*.csv`, SHA-256 per file) | - |
| `make_tts.py`, `make_latency_clips.py` | SYNTHETIC audio (Windows voices), always labelled synthetic | - |
| `set_config.py` | edits `firmware/sdkconfig` options in place (never touches the Wi-Fi credentials) | - |
| `run_final.ps1 -Label L -Level -22.7 -SpeechGain 0.58` | the full re-verification: idle CPU quiet + speech, synthetic latency x2, real test replay | DEVICE-ACOUSTIC |
| `run_trigger.ps1 -Label L -Level -16.7` | streaming path only: 18 loud detections -> board detection -> first send, audio lost, RAM peak | DEVICE-ACOUSTIC |
| `calibrate_level.py --gain 0.5` | received speech level at the board; keep it equal between runs you compare (the level/gain values above are for the board's current position) | - |

## Firmware variants (same sources, `firmware/sdkconfig` options)

* production (= `firmware/sdkconfig.defaults`): `KWS_PROFILE_OPS=n KWS_INJECT_TEST=n KWS_TELEMETRY_MS=0 KWS_PREROLL_MS=0
  KWS_MIC_SELECT=0 KWS_MIC_SPACING_MM=55 KWS_RAM_LIMIT_KB=256`. The status line's `peak + IRAM code N KB` is the strict RAM reading
  (`ram_strict_peak_kb` in the results).
* profiling: `python set_config.py KWS_PROFILE_OPS=y` -> `[prof]` lines (per-op model time, per-stage feature time)
* injection: `python set_config.py KWS_INJECT_TEST=y` -> no mics/Wi-Fi, audio from the serial port (921600 baud)

Build/flash: `idf.py reconfigure; ninja -C build -j 3; idf.py -p <port> -b 921600 flash` (restore production options after
a variant).

## Rules followed

* Threshold chosen on validation data only (`run_host.py sweep`: highest cutoff with real-validation TPR >= 95 %);
  the test set (`sets/real_test.csv`) is frozen and reported once per model/cutoff.
* Synthetic audio is used for extra test conditions only and is labelled; no synthetic audio went into training here.
* The real sets are recordings the board itself captured (one speaker, one room) and exist because an earlier model
  fired on them (selection bias toward "hard/known" utterances). They cannot support a false-accepts-per-hour claim.
