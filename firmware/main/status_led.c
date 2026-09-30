#include "status_led.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "led_strip.h"

static const char *TAG = "led";
static led_strip_handle_t s_strip;
static SemaphoreHandle_t s_lock;  // the LED is used from several tasks
static volatile int64_t s_off_at;  // status_led_flash(): time to switch off (0 = none)

void status_led_init(int gpio) {
    if (gpio < 0) return;
    s_lock = xSemaphoreCreateMutex();
    led_strip_config_t strip_config = {
        .strip_gpio_num = gpio,
        .max_leds = 1,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
        .flags = {.invert_out = false},
    };
    led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .mem_block_symbols = 0,
        .flags = {.with_dma = false},
    };
    if (led_strip_new_rmt_device(&strip_config, &rmt_config, &s_strip) != ESP_OK) {
        ESP_LOGW(TAG, "RGB LED not available on GPIO%d", gpio);
        s_strip = NULL;
        return;
    }
    led_strip_clear(s_strip);
}

void status_led_set(uint8_t r, uint8_t g, uint8_t b) {  // steady: cancels a pending flash
    s_off_at = 0;
    if (!s_strip || !s_lock) return;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    led_strip_set_pixel(s_strip, 0, r, g, b);
    led_strip_refresh(s_strip);
    xSemaphoreGive(s_lock);
}

void status_led_off(void) {
    s_off_at = 0;
    if (!s_strip || !s_lock) return;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    led_strip_clear(s_strip);
    xSemaphoreGive(s_lock);
}

void status_led_flash(uint8_t r, uint8_t g, uint8_t b, int ms) {
    status_led_set(r, g, b);
    s_off_at = esp_timer_get_time() + (int64_t)ms * 1000;
}

void status_led_poll(void) {
    const int64_t off_at = s_off_at;
    if (off_at && esp_timer_get_time() > off_at) {
        s_off_at = 0;
        status_led_off();
    }
}
