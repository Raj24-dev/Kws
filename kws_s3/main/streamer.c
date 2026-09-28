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
#include "wake_word.h"

static const char *TAG = "stream";

#define CHUNK_MS 20                                          // live audio: one 320-byte message per 20 ms
#define CHUNK_SAMPLES (AUDIO_SAMPLE_RATE * CHUNK_MS / 1000)
#define BURST_SAMPLES 960                                    // catching up (the pre-roll): up to 60 ms per message
#define SAMPLES_PER_MS (AUDIO_SAMPLE_RATE / 1000)
#define SEND_TIMEOUT_MS 1000  // a stall longer than the ring's slack loses audio anyway (counted: "audio lost ... net")
#define CONNECT_WAIT_MS 1000  // at a detection, how long to wait for a connection that is just coming up

static esp_websocket_client_handle_t s_ws;
static esp_transport_handle_t s_tcp;
static streamer_config_t s_cfg;
static TaskHandle_t s_task;
static atomic_bool s_active, s_connected, s_retrigger;
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

// Waits briefly for the WebSocket (e.g. right after boot), then sends the start message.
static bool open_utterance(char *msg, size_t cap) {
    for (int waited = 0; !atomic_load(&s_connected); waited += 50) {
        if (waited >= CONNECT_WAIT_MS) return false;
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    snprintf(msg, cap,
             "{\"type\":\"start\",\"device\":\"%s\",\"wake_word\":\"%s\",\"source\":\"%s\",\"score\":%.3f,"
             "\"sample_rate\":%d,\"format\":\"mulaw\",\"channels\":1,\"preroll_ms\":%d,\"chunk_ms\":%d}",
             s_device, s_cfg.wake_word, s_source, s_score, AUDIO_SAMPLE_RATE, s_cfg.preroll_ms, CHUNK_MS);
    return send_text(msg);
}

static void stream_task(void *arg) {
    static uint8_t buf[BURST_SAMPLES];
    char msg[320];

    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        const int64_t t_trigger = s_t_trigger;
        atomic_store(&s_retrigger, false);

        // Start preroll_ms before the detection (the ring buffer holds it; nothing is copied). 0 = at the detection:
        // only what is said after the wake word goes to the server.
        const uint32_t now = audio_ring_pos();
        uint32_t preroll = (uint32_t)s_cfg.preroll_ms * SAMPLES_PER_MS;
        if (preroll > now) preroll = now;
        uint32_t pos = now - preroll;
        const float threshold = fmaxf(audio_input_noise_floor() * 3.16f, 40.0f);  // background noise + 10 dB

        const bool open = open_utterance(msg, sizeof(msg));
        if (!open) ESP_LOGW(TAG, "no server connection: detection not streamed");  // the LED already blinked
        uint32_t sent = 0, live_base = preroll;
        int silence_ms = 0, first_latency_ms = -1;
        const char *reason = open ? "max_length" : "no_server";

        while (open) {
            if (!atomic_load(&s_connected)) {
                reason = "disconnected";
                break;
            }
            if (audio_ring_pos() - pos < CHUNK_SAMPLES) {  // wait for a full 20 ms of new audio
                vTaskDelay(pdMS_TO_TICKS(5));
                continue;
            }
            const size_t n = audio_ring_read(&pos, buf, BURST_SAMPLES);
            if (esp_websocket_client_send_bin(s_ws, (const char *)buf, (int)n, pdMS_TO_TICKS(SEND_TIMEOUT_MS)) < 0) {
                reason = "send_error";
                break;
            }
            if (first_latency_ms < 0) {
                first_latency_ms = (int)((esp_timer_get_time() - t_trigger) / 1000);
                ESP_LOGI(TAG, "first audio sent %d ms after the detection", first_latency_ms);
            }
            sent += (uint32_t)n;
            if (atomic_exchange(&s_retrigger, false)) {  // a new command in the same stream: full time again
                live_base = sent;
                silence_ms = 0;
            }

            // End-of-speech detection on the live part (after the pre-roll)
            if (sent > live_base) {
                int64_t ss = 0;
                for (size_t i = 0; i < n; i++) {
                    const int32_t v = mulaw_decode(buf[i]);
                    ss += v * v;
                }
                const float rms = sqrtf((float)ss / (float)n);
                const int live_ms = (int)((sent - live_base) / SAMPLES_PER_MS);
                silence_ms = (rms < threshold) ? silence_ms + (int)(n / SAMPLES_PER_MS) : 0;
                if (live_ms >= s_cfg.min_ms && silence_ms >= s_cfg.silence_ms) {
                    reason = "silence";
                    break;
                }
                if (live_ms >= s_cfg.max_ms) {
                    reason = "max_length";
                    break;
                }
            }
        }

        const int duration_ms = (int)(sent / SAMPLES_PER_MS);
        float peak = 0;
        int event_ms = 0;
        ww_last_event(&peak, &event_ms);
        if (open && atomic_load(&s_connected)) {
            snprintf(msg, sizeof(msg),
                     "{\"type\":\"end\",\"reason\":\"%s\",\"duration_ms\":%d,\"first_audio_latency_ms\":%d,"
                     "\"event_peak\":%.2f,\"event_ms\":%d}",
                     reason, duration_ms, first_latency_ms, peak, event_ms);
            send_text(msg);
        }
        ESP_LOGI(TAG, "stream finished (%s): %d ms of audio sent (%u KB)", reason, duration_ms, (unsigned)(sent / 1024));
        atomic_store(&s_active, false);
        if (atomic_exchange(&s_retrigger, false)) streamer_trigger(s_score, "wake_word");  // heard while closing
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
        .reconnect_timeout_ms = 1000,
        .network_timeout_ms = 5000,
        .ping_interval_sec = 2,        // a dead connection is noticed within ~20 s, not at the next detection
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
    bool expected = false;
    if (!atomic_compare_exchange_strong(&s_active, &expected, true)) {
        atomic_store(&s_retrigger, true);  // already streaming: the running stream is extended
        return true;
    }
    s_score = score;
    strlcpy(s_source, source, sizeof(s_source));
    s_t_trigger = esp_timer_get_time();
    xTaskNotifyGive(s_task);
    return true;
}

bool streamer_is_active(void) { return atomic_load(&s_active); }
bool streamer_is_connected(void) { return atomic_load(&s_connected); }
