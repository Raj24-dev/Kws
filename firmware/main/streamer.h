// Sends the audio after a detection to the speech-recognition (ASR) server over a WebSocket that stays open, so
// no connection setup delays the first audio.
// The model on the board decides; the server only transcribes what is said after the wake word.
// Protocol (server/server.js):
//   text   {"type":"start", ..., "format":"mulaw", "preroll_ms":0, "chunk_ms":20}   wake word detected
//   binary G.711 mu-law, 16 kHz mono, 1 byte per sample: the pre-roll first if any (as fast as the network
//          allows), then the live audio in 20 ms messages
//   text   {"type":"end", ..., "lost_ms":0, "resumes":0}   after silence or the time limit
// If the connection drops mid-command, the stream waits up to 5 s for it, sends a start with "source":"resume" and
// continues where it stopped (the server appends it to the same recording); audio older than the ring (~0.47 s)
// is skipped and reported as lost_ms.
// The server answers {"type":"response","text":...}: the transcript, after "end" (printed on the serial monitor).
#pragma once
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *uri;        // ws://host:port/path
    const char *wake_word;  // reported in the start message
    int preroll_ms;         // audio from before the detection (at most ~1 s: see AUDIO_RING_SAMPLES)
    int min_ms;             // stream at least this long
    int max_ms;             // never longer than this
    int silence_ms;         // stop after this much silence
} streamer_config_t;

esp_err_t streamer_init(const streamer_config_t *cfg);
// Starts one utterance. If already streaming, the running stream is extended so the new command is captured too.
// Returns false only before streamer_init.
bool streamer_trigger(float score, const char *source);
bool streamer_is_active(void);
bool streamer_is_connected(void);

#ifdef __cplusplus
}
#endif
