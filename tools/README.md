# Developer tools

Tools for watching the board while you test it. They talk to the board over its USB serial port and need
`pyserial` (it is already in the ESP-IDF Python environment). `pip install -r tools/requirements.txt` installs everything the tools need.

| tool | what it does |
|---|---|
| `serial_log.py <port> [--reset] [--minutes N]` | serial monitor that also saves a timestamped log to `logs/` (boot log with `--reset`, long stability runs with `--minutes`) |
| `dashboard.py [<port>] [--reset] [--host 0.0.0.0]` / `dashboard.cmd` | live web page at http://localhost:8090: wake-word score, CPU per core, RAM (lowest/highest), microphone level, board health, and a tester scorecard |
| `fake_device.py <recording.wav>` | plays a recording to the cloud server like the board streams it (see [`../server/README.md`](../server/README.md)) |
| `report.py [recordings_dir]` | HTML page with player, spectrogram and metadata for every saved command (default `server/recordings`) |
| `transcription.html` | live transcription page, served by the dashboard at http://localhost:8090/transcription |

## Live dashboard

The dashboard needs firmware built with `KWS_TELEMETRY_MS` > 0 (for example 100: one `@T ...` line per 100 ms).
It is 0 in `firmware/sdkconfig.defaults` because the telemetry itself costs about 4 % of core 0.

`dashboard.cmd` (Windows) finds the ESP-IDF Python and the board's port by itself. While the page is open the
dashboard owns the serial port, so press "Release port" on the page before flashing.

Scorecard: press Space on the page every time you say the wake word. A detection from 1 s before to 2.5 s after
the mark counts as correct, a mark without a detection is a miss, and a detection nobody marked is a false alarm
(click it to override). "Export CSV" saves the scorecard; the raw log goes to `logs/`.

## Live transcription

`transcription.html` connects to the cloud server's `ws://<host>:3000/live` feed and shows "listening" as soon
as the board detects the wake word, then the command once the speaker stops. It also lists the server's recent
utterances (the last 100 are reloaded after a server restart). Override the server with `?asr=ws://<host>:3000/live`.
