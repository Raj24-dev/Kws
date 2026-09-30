// Cloud server for the SIH wake word device (firmware/).
//
// Device protocol (WebSocket ws://<host>:3000/ws, see firmware/main/streamer.h):
//   text   {"type":"start","device":..,"wake_word":..,"score":..,"format":"mulaw","preroll_ms":..}  wake word detected
//   binary audio, 16 kHz mono: G.711 mu-law (1 byte/sample) with "format":"mulaw", else 16-bit PCM little-endian
//          (pre-roll first, then live audio in 20 ms messages)
//   text   {"type":"end","reason":..,"duration_ms":..,"first_audio_latency_ms":..,"lost_ms":..,"resumes":..}
//   A start with "source":"resume" is the rest of an utterance after the connection dropped: it is appended to
//   the interrupted recording, so the result is one file and one transcript, even across a server restart.
// Other clients can still use register / heartbeat / trigger / stop.
//
// The wake word is detected by the model on the device; the server does not check it again. The device sends what
// is said after it, every frame goes to the speech-to-text service (stt-services/main.py, WebSocket /stream) the
// moment it arrives, and after "end" it is transcribed ("Close the door.") and sent back: {"type":"response","text"}.
// Live view (tools/transcription.html): WebSocket ws://<host>:3000/live gets every utterance as it happens:
//   {"type":"start"} (detected on the device: listening), {"type":"final"} (the transcript).
// Every utterance is saved as recordings/<time>_<device>_<wake word>.wav plus one line in recordings/index.jsonl
// (tools/report.py turns them into an HTML page).
import Fastify from "fastify";
import websocket from "@fastify/websocket";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";

dotenv.config();

const PORT = Number(process.env.PORT) || 3000;
const RECORDINGS_DIR = path.resolve(process.env.RECORDINGS_DIR || "recordings");
const SAMPLE_RATE = 16000;
const MAX_AUDIO_BYTES = 20 * SAMPLE_RATE * 2; // 20 s. The device sends at most 1 s pre-roll + 8 s per command.
const HEARTBEAT_TIMEOUT_MS = 60000;
const RESUME_WINDOW_MS = 30000; // an interrupted utterance waits this long for the device to resume it
const STT_STREAM_URL = (process.env.STT_URL || "http://127.0.0.1:8000").replace(/^http/, "ws") + "/stream";
const STT_FINAL_TIMEOUT_MS = 20000;

fs.mkdirSync(RECORDINGS_DIR, { recursive: true });

const pad = (n, w = 2) => String(n).padStart(w, "0");
function localTime(d = new Date()) {
  const date = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
  const time = `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return {
    iso: `${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6)}T${time.slice(0, 2)}:${time.slice(2, 4)}:${time.slice(4)}`,
    stamp: `${date}_${time}_${pad(d.getMilliseconds(), 3)}`,
  };
}
const log = (msg) => console.log(`[${new Date().toLocaleTimeString()}] ${msg}`);

function wavFile(pcm) {
  const h = Buffer.alloc(44);
  h.write("RIFF", 0);
  h.writeUInt32LE(36 + pcm.length, 4);
  h.write("WAVEfmt ", 8);
  h.writeUInt32LE(16, 16);
  h.writeUInt16LE(1, 20); // PCM
  h.writeUInt16LE(1, 22); // mono
  h.writeUInt32LE(SAMPLE_RATE, 24);
  h.writeUInt32LE(SAMPLE_RATE * 2, 28);
  h.writeUInt16LE(2, 32);
  h.writeUInt16LE(16, 34);
  h.write("data", 36);
  h.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([h, pcm]);
}

// G.711 mu-law -> 16-bit PCM (same decoder as firmware/main/audio_input.h)
const MULAW = Int16Array.from({ length: 256 }, (_, i) => {
  const u = ~i & 0xff;
  const t = (((u & 0x0f) << 3) + 0x84) << ((u & 0x70) >> 4);
  return u & 0x80 ? 0x84 - t : t - 0x84;
});
function mulawToPcm(buf) {
  const pcm = Buffer.alloc(buf.length * 2);
  for (let i = 0; i < buf.length; i++) pcm.writeInt16LE(MULAW[buf[i]], i * 2);
  return pcm;
}

function levels(pcm) {
  let sum = 0;
  let peak = 0;
  for (let i = 0; i + 1 < pcm.length; i += 2) {
    const s = pcm.readInt16LE(i);
    sum += s * s;
    peak = Math.max(peak, Math.abs(s));
  }
  const n = pcm.length >> 1;
  const db = (v) => (v > 0 ? +(20 * Math.log10(v / 32768)).toFixed(1) : -120);
  return { rms_dbfs: db(n ? Math.sqrt(sum / n) : 0), peak_dbfs: db(peak) };
}

// ---- Live view: every page connected to /live gets the utterance events; a new page first gets the recent ones
const liveClients = new Set();
const liveHistory = []; // recent events, replayed to a page that connects
// The transcripts stay on the server: after a restart the recent ones come back from index.jsonl
try {
  for (const line of fs.readFileSync(path.join(RECORDINGS_DIR, "index.jsonl"), "utf8").trim().split("\n").slice(-100)) {
    let r;
    try { r = JSON.parse(line); } catch { continue; } // a line cut short by a crash
    const id = r.file.replace(/\.wav$/, ""), T = new Date(r.time).getTime() / 1000;
    liveHistory.push(
      JSON.stringify({ type: "start", id, device: r.device, wake_word: r.wake_word, score: r.score, source: r.source, T }),
      JSON.stringify({ type: "final", id, device: r.device, text: r.transcript, verified: r.verified, verdict: r.verdict ?? r.verified,
                       verdict_ms: r.verdict_ms ?? null, stt_final_ms: r.stt_final_ms ?? null, duration_s: r.duration_s,
                       reason: r.reason, file: r.file, T: T + r.duration_s }));
  }
} catch {} // no recordings yet
function live(obj) {
  const msg = JSON.stringify({ ...obj, T: Date.now() / 1000 });
  liveHistory.push(msg);
  while (liveHistory.length > 400) liveHistory.shift();
  for (const c of liveClients) if (c.readyState === 1) c.send(msg);
}

// ---- Live link to the speech-to-text service for one utterance. Frames are forwarded as they arrive (queued only
// ---- until the link is open). finish() ends the utterance and resolves with the transcript (or null if the
// ---- service is unreachable). The pre-roll, if the device sends one, is not transcribed.
function openStt(info) {
  const ws = new WebSocket(STT_STREAM_URL);
  const queue = [JSON.stringify({ type: "start", preroll_ms: info.preroll_ms ?? 0 })];
  let resolveFinal;
  const final = new Promise((resolve) => (resolveFinal = resolve));
  ws.onopen = () => queue.splice(0).forEach((m) => ws.send(m));
  ws.onmessage = (ev) => {
    try {
      const m = JSON.parse(ev.data);
      if (m.type === "final") resolveFinal(m.text);
    } catch {} // not JSON: a throw here would take the whole server down
  };
  ws.onerror = () => resolveFinal(null);
  ws.onclose = () => resolveFinal(null);
  const send = (data) => (ws.readyState === WebSocket.OPEN ? ws.send(data) : queue.push(data));
  return {
    send,
    finish() {
      if (ws.readyState <= WebSocket.OPEN) send(JSON.stringify({ type: "end" }));
      const timeout = setTimeout(() => resolveFinal(null), STT_FINAL_TIMEOUT_MS);
      return final.finally(() => {
        clearTimeout(timeout);
        ws.close();
      });
    },
    close: () => ws.close(),
  };
}

// ---- Utterances in progress: the audio is appended to <name>.part as it arrives (survives a crash), the start
// ---- message is kept in <name>.part.json. finishUtterance() turns them into <name>.wav.
function openUtterance(deviceId, info) {
  const t = localTime();
  const base = `${t.stamp}_${deviceId}_${info.wake_word || "utt"}`.replace(/[^\w.-]/g, "_");
  const part = path.join(RECORDINGS_DIR, `${base}.part`);
  fs.writeFileSync(`${part}.json`, JSON.stringify({ deviceId, info, time: t.iso }));
  return { deviceId, info, time: t.iso, base, part, fd: fs.openSync(part, "a"), bytes: 0, truncated: false, resumed: 0,
           startedAt: Date.now(), firstAudioMs: null, stt: null };
}

function loadPart(part) {
  const meta = JSON.parse(fs.readFileSync(`${part}.json`, "utf8"));
  return { ...meta, base: path.basename(part, ".part"), part, fd: null, bytes: fs.statSync(part).size, truncated: false,
           resumed: 0, stt: null };
}

function discardUtterance(utt) {
  utt.stt?.close();
  if (utt.fd !== null) fs.closeSync(utt.fd);
  fs.rmSync(utt.part, { force: true });
  fs.rmSync(`${utt.part}.json`, { force: true });
}

// Interrupted utterances wait for a resume; if none comes they are saved as they are. The utterance keeps its live
// speech-to-text link while it waits, so a resumed one is transcribed as a whole.
const pendingParts = new Map(); // .part path -> { utt, timer }
function awaitResume(utt) {
  const timer = setTimeout(() => {
    pendingParts.delete(utt.part);
    finishUtterance(utt, { reason: "connection_lost" }).catch((err) => log(`could not save: ${err.message}`));
  }, RESUME_WINDOW_MS);
  pendingParts.set(utt.part, { utt, timer });
}

// The device's newest interrupted utterance that is still waiting (part names start with the time)
function takePending(deviceId) {
  let newest = null;
  for (const [part, p] of pendingParts) if (p.utt.deviceId === deviceId && (!newest || part > newest)) newest = part;
  if (!newest) return null;
  const { utt, timer } = pendingParts.get(newest);
  clearTimeout(timer);
  pendingParts.delete(newest);
  return utt;
}

let sttWarned = false;

// Saves one utterance, transcribes it and records it in index.jsonl. Returns the index record.
async function finishUtterance(utt, end) {
  if (utt.fd !== null) fs.closeSync(utt.fd);
  utt.fd = null;
  const pcm = fs.readFileSync(utt.part);
  const file = `${utt.base}.wav`;
  fs.writeFileSync(path.join(RECORDINGS_DIR, file), wavFile(pcm));
  fs.rmSync(utt.part, { force: true });
  fs.rmSync(`${utt.part}.json`, { force: true });

  const record = {
    time: utt.time,
    file,
    device: utt.deviceId,
    wake_word: utt.info.wake_word ?? null,
    source: utt.info.source ?? utt.info.type,
    score: utt.info.score ?? null,
    preroll_ms: utt.info.preroll_ms ?? null,
    duration_s: +(pcm.length / 2 / SAMPLE_RATE).toFixed(2),
    reason: end.reason ?? (end.type === "stop" ? "stop" : null),
    first_audio_latency_ms: end.first_audio_latency_ms ?? null,
    first_audio_rx_ms: utt.firstAudioMs ?? null, // server: start message -> first audio frame
    start_rx_epoch_ms: utt.startRxEpochMs ?? null, // arrival times on this PC's clock (latency benchmark)
    first_audio_rx_epoch_ms: utt.firstAudioEpochMs ?? null,
    resumes: end.resumes ?? utt.resumed, // reconnects bridged within this utterance
    lost_ms: end.lost_ms ?? null, // audio the device could not deliver (network too slow for its buffer)
    event_peak: end.event_peak ?? null, // the device's wake word score peak and length
    event_ms: end.event_ms ?? null,
    truncated: utt.truncated,
    ...levels(pcm),
    transcript: null, // what was said after the wake word
  };
  const t0 = Date.now();
  record.transcript = utt.stt ? await utt.stt.finish() : null;
  if (record.transcript === null && pcm.length) { // no live link (recovered after a restart) or it broke: send the file
    const stt = openStt(utt.info);
    stt.send(pcm);
    record.transcript = await stt.finish();
  }
  record.stt_final_ms = Date.now() - t0; // "end" -> transcript
  if (record.transcript !== null) {
    log(`${utt.deviceId}: "${record.transcript}" (${record.stt_final_ms} ms after the end)`);
    sttWarned = false;
  } else if (!sttWarned) {
    log(`speech-to-text unavailable (${STT_STREAM_URL}); saving audio only. Start stt-services/main.py.`);
    sttWarned = true;
  }
  fs.appendFileSync(path.join(RECORDINGS_DIR, "index.jsonl"), JSON.stringify(record) + "\n");
  live({ type: "final", id: utt.base, device: utt.deviceId, text: record.transcript, score: record.score,
         first_audio_rx_ms: record.first_audio_rx_ms, stt_final_ms: record.stt_final_ms, duration_s: record.duration_s,
         reason: record.reason, file });
  log(`${utt.deviceId}: saved ${file} (${record.duration_s} s, ${record.reason}` +
      (utt.resumed ? `, stitched across ${utt.resumed} reconnect(s)` : "") + `, rms ${record.rms_dbfs} dBFS)`);
  return record;
}

// Interrupted utterances left over from before a server restart get the same chance to be resumed
for (const f of fs.readdirSync(RECORDINGS_DIR).filter((f) => f.endsWith(".part"))) {
  const part = path.join(RECORDINGS_DIR, f);
  try {
    awaitResume(loadPart(part));
    log(`interrupted utterance ${f} waits ${RESUME_WINDOW_MS / 1000} s for its device to resume`);
  } catch (err) {
    log(`cannot recover ${f}: ${err.message}`);
  }
}

const app = Fastify();
await app.register(websocket);

app.get("/health", async () => ({ status: "running" }));

const devices = new Map();

// Live view for the dashboard (tools/dashboard.html). Any origin may connect: it only reads.
app.get("/live", { websocket: true }, (socket) => {
  liveClients.add(socket);
  socket.send(JSON.stringify({ type: "hello", history: liveHistory.map((m) => JSON.parse(m)) }));
  socket.on("close", () => liveClients.delete(socket));
  socket.on("error", () => liveClients.delete(socket));
});

app.get("/ws", { websocket: true }, (socket, req) => {
  const peer = req.socket.remoteAddress;
  let deviceId = null;
  let utt = null; // utterance being received
  let lastSeen = Date.now();
  log(`client connected from ${peer}`);

  const send = (obj) => socket.readyState === 1 && socket.send(JSON.stringify(obj));
  const register = (id, firmware) => {
    deviceId = id;
    devices.set(id, { socket, firmware, connectedAt: Date.now() });
    log(`registered: ${id}`);
  };

  // The device's WebSocket client pings every 10 s: any traffic counts as a heartbeat.
  socket.on("ping", () => (lastSeen = Date.now()));
  const heartbeat = setInterval(() => {
    if (Date.now() - lastSeen > HEARTBEAT_TIMEOUT_MS) {
      log(`heartbeat timeout: ${deviceId || peer}`);
      socket.close(1000, "Heartbeat timeout");
    }
  }, 30000);

  socket.on("message", async (message, isBinary) => {
    lastSeen = Date.now();
    const rxEpochMs = performance.timeOrigin + performance.now(); // arrival time, sub-ms (latency benchmark)

    // ---------- Binary audio ----------
    if (isBinary) {
      if (!utt || utt.truncated) return; // audio outside an utterance, or after the size limit
      const pcm = utt.info.format === "mulaw" ? mulawToPcm(message) : message;
      if (utt.bytes + pcm.length > MAX_AUDIO_BYTES) {
        log(`${deviceId}: utterance longer than ${MAX_AUDIO_BYTES / 32000} s, keeping the first part`);
        send({ type: "error", message: "Maximum stream duration exceeded", action: "stop-stream" });
        utt.truncated = true;
        return;
      }
      utt.stt?.send(pcm); // to the ASR first: this is the latency that counts
      if (utt.firstAudioMs == null && utt.startedAt) { // (an utterance recovered after a restart has no start time)
        utt.firstAudioMs = Date.now() - utt.startedAt;
        utt.firstAudioEpochMs = rxEpochMs;
        live({ type: "first_audio", id: utt.base, rx_epoch_ms: rxEpochMs, bytes: message.length });
      }
      try {
        fs.writeSync(utt.fd, pcm); // synchronous 640 B write per 20 ms: fine for a few devices
      } catch (err) { // disk full etc.: keep transcribing (a throw here would stop the whole server)
        if (!utt.writeFailed) log(`${deviceId}: cannot save audio: ${err.message}`);
        utt.writeFailed = true;
      }
      utt.bytes += pcm.length;
      return;
    }

    // ---------- JSON messages ----------
    let data;
    try {
      data = JSON.parse(message.toString());
    } catch {
      send({ type: "error", message: "Invalid JSON" });
      return;
    }

    try {
      switch (data.type) {
        case "register":
          if (deviceId) {
            send({ type: "ack", status: "already-registered", deviceId });
            break;
          }
          if (typeof data.deviceId !== "string" || data.deviceId.trim() === "") {
            send({ type: "error", message: "Invalid deviceId" });
            break;
          }
          register(data.deviceId, data.firmware);
          send({ type: "ack", status: "registered", deviceId });
          break;

        case "heartbeat":
          send({ type: "ack", status: "alive" });
          break;

        case "start": // the ESP32 firmware: wake word detected (registers itself with its MAC address)
        case "trigger": { // other clients, after "register"
          if (!deviceId && data.type === "start" && typeof data.device === "string" && data.device) register(data.device);
          if (!deviceId) {
            send({ type: "error", message: "Register first" });
            break;
          }
          if (utt) {
            send({ type: "ack", status: "already-streaming" });
            break;
          }
          const pending = data.source === "resume" ? takePending(deviceId) : null;
          if (pending) { // continue the interrupted recording and its speech-to-text link
            utt = pending;
            utt.fd = fs.openSync(utt.part, "a");
            utt.resumed++;
            log(`${deviceId}: resumed after a reconnect, appending to ${path.basename(utt.part)}`);
          } else {
            utt = openUtterance(deviceId, data);
            utt.startRxEpochMs = rxEpochMs;
            log(`${deviceId}: ${data.type === "start" ? `wake word "${data.wake_word}" (score ${data.score}${data.source === "resume" ? ", resume without the first part" : ""})` : "trigger"}`);
            const current = utt;
            live({ type: "start", id: current.base, device: deviceId, wake_word: data.wake_word ?? null,
                   score: data.score ?? null, source: data.source ?? data.type });
            current.stt = openStt(data);
          }
          if (data.type === "trigger") send({ type: "ack", status: "ready-for-audio" });
          break;
        }

        case "end": // the ESP32 firmware: silence or time limit reached
        case "stop": {
          const current = utt;
          utt = null; // a new utterance may start while this one is being transcribed
          if (!current || current.bytes === 0) {
            if (current) discardUtterance(current);
            send({ type: "error", message: "No audio received" });
            break;
          }
          const record = await finishUtterance(current, data);
          send({ type: "response", text: record.transcript, saved: record.file });
          break;
        }

        default:
          send({ type: "error", message: "Unknown message type" });
      }
    } catch (err) {
      log(`${deviceId || peer}: error handling "${data.type}": ${err.message}`);
      send({ type: "error", message: "Server error" });
    }
  });

  socket.on("close", () => {
    clearInterval(heartbeat);
    if (utt) {
      try {
        if (utt.bytes > 0) {
          fs.closeSync(utt.fd);
          utt.fd = null;
          awaitResume(utt); // keep what arrived; the device resumes it after reconnecting
          log(`${deviceId}: connection lost mid-utterance, waiting ${RESUME_WINDOW_MS / 1000} s for a resume`);
        } else {
          discardUtterance(utt);
        }
      } catch (err) {
        log(`could not keep the interrupted utterance: ${err.message}`);
      }
    }
    utt = null;
    if (deviceId && devices.get(deviceId)?.socket === socket) devices.delete(deviceId);
    log(`disconnected: ${deviceId || peer}`);
  });

  socket.on("error", (err) => log(`WebSocket error: ${err.message}`));
});

await app.listen({ port: PORT, host: "0.0.0.0" });
log(`Server running on port ${PORT}: device URI ws://<this PC's IP>:${PORT}/ws, recordings in ${RECORDINGS_DIR}`);
