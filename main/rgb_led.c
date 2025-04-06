//rgb_led.c
#include "rgb_led.h"
#include "led_strip.h"
#include "esp_err.h"

#define RGB_LED_GPIO 48  // Уточни пин, если другой
#define RGB_LED_NUM 1

static led_strip_handle_t led_strip;

void rgb_led_init(void) {
    led_strip_config_t strip_config = {
        .strip_gpio_num = RGB_LED_GPIO,
        .max_leds = RGB_LED_NUM,
        .led_model = LED_MODEL_WS2812,
        .flags.invert_out = false,
    };

    led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000, // 10 MHz
        .flags.with_dma = false,
    };

    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &led_strip));
    rgb_led_off();
}

void rgb_led_set_color(uint8_t r, uint8_t g, uint8_t b) {
    ESP_ERROR_CHECK(led_strip_set_pixel(led_strip, 0, r, g, b));
    ESP_ERROR_CHECK(led_strip_refresh(led_strip));
}

// Пресеты
void rgb_led_red(void)    { rgb_led_set_color(255, 0, 0); }
void rgb_led_green(void)  { rgb_led_set_color(0, 255, 0); }
void rgb_led_blue(void)   { rgb_led_set_color(0, 0, 255); }
void rgb_led_white(void)  { rgb_led_set_color(255, 255, 255); }
void rgb_led_off(void)    { rgb_led_set_color(0, 0, 0); }