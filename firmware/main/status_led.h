// RGB status LED (the WS2812 LED on ESP32-S3 DevKit boards)
//   green   = wake word confirmed by the server (on while the command is streamed), or a short blink when there
//             is no server to confirm it
//   red     = short blink: the server rejected the detection (false trigger); steady at boot: fatal problem
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void status_led_init(int gpio);  // gpio < 0 disables the LED
void status_led_set(uint8_t r, uint8_t g, uint8_t b);
void status_led_off(void);
void status_led_flash(uint8_t r, uint8_t g, uint8_t b, int ms);  // on for ms, then off (see status_led_poll)
void status_led_poll(void);  // call every ~20 ms: switches a flash off when its time is up

#ifdef __cplusplus
}
#endif
