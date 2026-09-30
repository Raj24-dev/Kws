#include "streamer.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "audio_input.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "esp_transport.h"
#include "esp_transport_tcp.h"
#include "esp_transport_ws.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "status_led.h"
#include "wake_word.h"

static const char *TAG = "stream";

#define CHUNK_MS 20                                          // live audio: one 320-byte message per 20 ms
#define CHUNK_SAMPLES (AUDIO_SAMPLE_RATE * CHUNK_MS / 1000)
#define BURST_SAMPLES 960                                    // catching up (pre-roll, after a stall): up to 60 ms per message
#define SAMPLES_PER_MS (AUDIO_SAMPLE_RATE / 1000)
// A send that times out makes the client library drop the connection; the stream then resumes on a new one. Kept
// short on purpose: a send that started just as the connection broke holds the library's lock for this long and
// delays its reconnect by as much (measured with 4 s: every drop cost 4 s instead of ~1 s).
#define SEND_TIMEOUT_MS 1000
#define CONNECT_WAIT_MS 1000  // at a detection, how long to wait for a connection that is just coming up
#define RESUME_WAIT_MS 5000   // after a drop mid-command, how long to wait for the connection to come back
#define MAX_RESUMES 3         // per utterance
#define RECONNECT_MS 1000     // reconnect attempts while idle; every 200 ms while a command waits for the connection

static esp_websocket_client_handle_t s_ws;
static esp_transport_handle_t s_tcp;
static streamer_config_t s_cfg;
static TaskHandle_t s_task;
static atomic_bool s_active, s_connected, s_retrigger;
static atomic_uint s_trigger_pos;  // ring position of the latest detection: streams start there even if they start late
static float s_score;
static char s_source[16];
static int64_t s_t_trigger;
static char s_device[18];
static bool s_error_shown;  // one warning per outage, not one per reconnect attempt (every second)

static void on_server_text(const char *text, int len) {  // the transcript of the command, for the log
    ESP_LOGI(TAG, "server: %.*s", len, text);
}

static void ws_event(void *arg, esp_event_base_t base, int32_t id, void *event_data) {
    const esp_websocket_event_data_t *d = (const esp_websocket_event_data_t *)event_data;
    switch (id) {
        case WEBSOCKET_EVENT_CONNECTED: {
            // Send every audio message at once. The WebSocket layer writes a frame's header and payload separately;
            // with Nagle's algorithm the payload would wait until the server acknowledged the header, and Windows
            // delays that acknowledgement by up to 200 ms.
            const int one = 1, fd = esp_transport_get_socket(s_tcp);
            if (fd < 0 || setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0)
                ESP_LOGW(TAG, "could not disable Nagle (TCP_NODELAY): audio may arrive in bursts");
            atomic_store(&s_connected, true);
            s_error_shown = false;
            ESP_LOGI(TAG, "connected to the server %s", s_cfg.uri);
            break;
        }
        case WEBSOCKET_EVENT_DISCONNECTED:
            if (atomic_exchange(&s_connected, false))
                ESP_LOGW(TAG, "server disconnected (error type %d, errno %d, close code %d), retrying...",
                         d ? (int)d->error_handle.error_type : -1, d ? d->error_handle.esp_transport_sock_errno : -1,
                         d ? d->close_status_code : -1);
            break;
        case WEBSOCKET_EVENT_DATA:  // text message from the server (the transcript)
            if (d && d->op_code == 0x1 && d->data_len > 0 && d->payload_offset == 0 && d->data_len == d->payload_len)
                on_server_text((const char *)d->data_ptr, d->data_len);
            break;
        case WEBSOCKET_EVENT_ERROR:  // the data is the client library's error message (its own log is muted above)
            if (atomic_load(&s_connected))
                ESP_LOGW(TAG, "connection broke: %.*s", d ? d->data_len : 0, d && d->data_ptr ? d->data_ptr : "");
            else if (!s_error_shown)
                ESP_LOGW(TAG, "cannot reach the server %s (is it running? same network?) - retrying every second", s_cfg.uri);
            s_error_shown = true;
            break;
        default:
            break;
    }
}

static bool send_text(const char *msg) {
    return esp_websocket_client_send_text(s_ws, msg, (int)strlen(msg), pdMS_TO_TICKS(SEND_TIMEOUT_MS)) >= 0;
}

// Waits up to wait_ms for the WebSocket (e.g. right after boot, or after a drop), retrying every 200 ms meanwhile:
// every second of waiting is audio the ring buffer may not be able to keep
static bool wait_connected(int wait_ms) {
    if (atomic_load(&s_connected)) return true;
    esp_websocket_client_set_reconnect_timeout(s_ws, 200);
    for (int waited = 0; !atomic_load(&s_connected) && waited < wait_ms; waited += 50) vTaskDelay(pdMS_TO_TICKS(50));
    esp_websocket_client_set_reconnect_timeout(s_ws, RECONNECT_MS);
    return atomic_load(&s_connected);
}

// source "resume": the rest of an utterance whose connection dropped (the server appends it to the same recording)
static bool send_start(char *msg, size_t cap, const char *source) {
    snprintf(msg, cap,
             "{\"type\":\"start\",\"device\":\"%s\",\"wake_word\":\"%s\",\"source\":\"%s\",\"score\":%.3f,"
             "\"sample_rate\":%d,\"format\":\"mulaw\",\"channels\":1,\"preroll_ms\":%d,\"chunk_ms\":%d}",
             s_device, s_cfg.wake_word, source, s_score, AUDIO_SAMPLE_RATE, s_cfg.preroll_ms, CHUNK_MS);
    return send_text(msg);
}

static void stream_task(void *arg) {
    static uint8_t buf[BURST_SAMPLES];
    char msg[360];
    bool again = false;  // a wake word heard while the previous stream was closing: stream again at once

    for (;;) {
        if (!again) ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        again = false;
        const int64_t t_trigger = s_t_trigger;
        atomic_store(&s_retrigger, false);

        // From preroll_ms before the detection (the ring buffer holds it; nothing is copied). 0 = at the detection:
        // only what is said after the wake word goes to the server.
        const uint32_t det = atomic_load(&s_trigger_pos);
        uint32_t preroll = (uint32_t)s_cfg.preroll_ms * SAMPLES_PER_MS;
        if (preroll > det) preroll = det;
        const uint32_t start = det - preroll;
        uint32_t pos = start, live = det;  // live: where the current command starts (after the pre-roll)
        const float threshold = fmaxf(audio_input_noise_floor() * 3.16f, 40.0f);  // background noise + 10 dB

        const bool open = wait_connected(CONNECT_WAIT_MS) && send_start(msg, sizeof(msg), s_source);
        if (!open) ESP_LOGW(TAG, "no server connection: detection not streamed");
        uint32_t sent = 0, lost = 0;
        int silence_ms = 0, first_latency_ms = -1, resumes = 0;
        const char *reason = open ? "max_length" : "no_server";

        while (open) {
            if (!atomic_load(&s_connected)) {  // dropped mid-command: continue the same utterance when it is back
                if (resumes >= MAX_RESUMES || !wait_connected(RESUME_WAIT_MS) || !send_start(msg, sizeof(msg), "resume")) {
                    reason = "disconnected";
                    break;
                }
                resumes++;
                ESP_LOGI(TAG, "connection back: stream resumed");
                continue;
            }
            if (audio_ring_pos() - pos < CHUNK_SAMPLES) {  // wait for a full 20 ms of new audio
                vTaskDelay(pdMS_TO_TICKS(5));
                continue;
            }
            const uint32_t from = pos;
            const size_t n = audio_ring_read(&pos, buf, BURST_SAMPLES);
            lost += pos - (uint32_t)n - from;  // skipped: older than the ring could keep
            if (esp_websocket_client_send_bin(s_ws, (const char *)buf, (int)n, pdMS_TO_TICKS(SEND_TIMEOUT_MS)) < 0) {
                pos -= (uint32_t)n;  // not delivered: sent again after the reconnect (while the ring still has it)
                vTaskDelay(pdMS_TO_TICKS(CHUNK_MS));  // the disconnect event may still be on its way
                continue;
            }
            if (first_latency_ms < 0) {
                first_latency_ms = (int)((esp_timer_get_time() - t_trigger) / 1000);
                ESP_LOGI(TAG, "first audio sent %d ms after the detection", first_latency_ms);
            }
            sent += (uint32_t)n;
            if (atomic_exchange(&s_retrigger, false)) {  // a new command in the same stream: full time again
                live = atomic_load(&s_trigger_pos);
                silence_ms = 0;
            }

            // End-of-speech detection on the live part (after the pre-roll)
            if ((int32_t)(pos - live) > 0) {
                int64_t ss = 0;
                for (size_t i = 0; i < n; i++) {
                    const int32_t v = mulaw_decode(buf[i]);
                    ss += v * v;
                }
                const float rms = sqrtf((float)ss / (float)n);
                const int live_ms = (int)((pos - live) / SAMPLES_PER_MS);
                silence_ms = (rms < threshold) ? silence_ms + (int)(n / SAMPLES_PER_MS) : 0;
                if (live_ms >= s_cfg.min_ms && silence_ms >= s_cfg.silence_ms) {
                    reason = "silence";
                    break;
                }
                // Repeated wake words extend a stream, but never past twice the limit
                if (live_ms >= s_cfg.max_ms || (int)((pos - start) / SAMPLES_PER_MS) >= 2 * s_cfg.max_ms) {
                    reason = "max_length";
                    break;
                }
            }
        }

        const int duration_ms = (int)(sent / SAMPLES_PER_MS), lost_ms = (int)(lost / SAMPLES_PER_MS);
        float peak = 0;
        int event_ms = 0;
        ww_last_event(&peak, &event_ms);
        if (open && atomic_load(&s_connected)) {
            snprintf(msg, sizeof(msg),
                     "{\"type\":\"end\",\"reason\":\"%s\",\"duration_ms\":%d,\"first_audio_latency_ms\":%d,"
                     "\"lost_ms\":%d,\"resumes\":%d,\"event_peak\":%.2f,\"event_ms\":%d}",
                     reason, duration_ms, first_latency_ms, lost_ms, resumes, peak, event_ms);
            send_text(msg);
        }
        if (!open || strcmp(reason, "disconnected") == 0) status_led_flash(60, 0, 0, 600);  // red: not delivered
        ESP_LOGI(TAG, "stream finished (%s): %d ms of audio sent (%u KB), %d ms lost, %d resumes", reason, duration_ms,
                 (unsigned)(sent / 1024), lost_ms, resumes);
        atomic_store(&s_active, false);
        // A wake word heard while closing only set the flag (streamer_trigger saw an active stream): start from it now
        again = atomic_exchange(&s_retrigger, false) && !atomic_exchange(&s_active, true);
    }
}

esp_err_t streamer_init(const streamer_config_t *cfg) {
    s_cfg = *cfg;
    uint8_t mac[6] = {0};
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(s_device, sizeof(s_device), "%02x%02x%02x%02x%02x%02x", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    // Our own TCP + WebSocket transport (instead of the client's), so the socket can be tuned when it connects
    s_tcp = esp_transport_tcp_init();
    esp_transport_handle_t ws = s_tcp ? esp_transport_ws_init(s_tcp) : NULL;
    if (!ws) return ESP_ERR_NO_MEM;
    const char *host = strstr(cfg->uri, "://");
    const char *path = strchr(host ? host + 3 : cfg->uri, '/');
    const esp_transport_ws_config_t ws_tcfg = {.ws_path = path ? path : "/", .propagate_control_frames = true};
    esp_transport_ws_set_config(ws, &ws_tcfg);

    esp_websocket_client_config_t ws_cfg = {
        .uri = cfg->uri,
        .ext_transport = ws,
        .buffer_size = 1024,           // >= BURST_SAMPLES: one audio message = one WebSocket frame
        .task_stack = 3584,          // uses ~2.6 KB
        .reconnect_timeout_ms = RECONNECT_MS,
        .network_timeout_ms = 5000,
        .ping_interval_sec = 5,        // a dead connection is noticed within ~25 s, not at the next detection
        .pingpong_timeout_sec = 20,    // the phone hotspot stalls for >6 s at times (measured): 6 s dropped live streams
    };
    // The client library logs 4 error lines per failed attempt (every second while the server is down)
    esp_log_level_set("websocket_client", ESP_LOG_NONE);
    esp_log_level_set("transport_ws", ESP_LOG_NONE);
    esp_log_level_set("transport_base", ESP_LOG_NONE);
    esp_log_level_set("esp-tls", ESP_LOG_NONE);
    s_ws = esp_websocket_client_init(&ws_cfg);
    if (!s_ws) return ESP_FAIL;
    esp_websocket_register_events(s_ws, WEBSOCKET_EVENT_ANY, ws_event, NULL);
    esp_err_t err = esp_websocket_client_start(s_ws);
    if (err != ESP_OK) return err;
    if (xTaskCreatePinnedToCore(stream_task, "streamer", 3584, NULL, 8, &s_task, 0) != pdPASS) return ESP_ERR_NO_MEM;
    return ESP_OK;
}

bool streamer_trigger(float score, const char *source) {
    if (!s_task) return false;
    // Where and when the wake word was detected: the stream starts there even if the stream task gets to it late
    atomic_store(&s_trigger_pos, audio_ring_pos());
    s_t_trigger = esp_timer_get_time();
    s_score = score;
    strlcpy(s_source, source, sizeof(s_source));
    bool expected = false;
    if (!atomic_compare_exchange_strong(&s_active, &expected, true)) {
        atomic_store(&s_retrigger, true);  // already streaming: the running stream is extended
        return true;
    }
    xTaskNotifyGive(s_task);
    return true;
}

bool streamer_is_active(void) { return atomic_load(&s_active); }
bool streamer_is_connected(void) { return atomic_load(&s_connected); }
