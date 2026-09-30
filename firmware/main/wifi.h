// Wi-Fi station: connects in the background and reconnects automatically.
#pragma once
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t wifi_start(const char *ssid, const char *password);
bool wifi_is_connected(void);

#ifdef __cplusplus
}
#endif
