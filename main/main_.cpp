// File: main/main.cpp

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "mfcc_api.h"
#include "lstm_api.h"
#include <cmath>
#include <cstring>

#define TAG "MAIN"
#define SAMPLE_SIZE 1024
#define FRAME_LEN   256
#define MFCC_NUM    13
#define LSTM_TIMESTEPS 99

// Массив данных в DRAM
static uint8_t dram_data[SAMPLE_SIZE];
static float g_audio_frame[FRAME_LEN];

// Функция для записи данных в DRAM
void write_to_dram(uint8_t *data, size_t size) {
    if (size > SAMPLE_SIZE) {
        ESP_LOGE(TAG, "Data size exceeds DRAM buffer size");
        return;
    }
    memcpy(dram_data, data, size);
}

// Функция для чтения данных из DRAM
void read_from_dram(uint8_t *data, size_t size) {
    if (size > SAMPLE_SIZE) {
        ESP_LOGE(TAG, "Data size exceeds DRAM buffer size");
        return;
    }
    memcpy(data, dram_data, size);
}

// Синусоидальный сигнал для теста MFCC
void generate_sine_wave(float *buffer, size_t len, float freq, float rate) {
    for (size_t i = 0; i < len; ++i) {
        buffer[i] = sinf(2.0f * M_PI * freq * i / rate);
    }
}

// Функция для выполнения MFCC и копирования в PSRAM
void run_mfcc_test() {
    ESP_LOGI(TAG, "=== BEGIN MFCC TEST ===");

    generate_sine_wave(g_audio_frame, FRAME_LEN, 440.0f, 16000.0f);

    if (!mfcc_init()) {
        ESP_LOGE(TAG, "MFCC init failed");
        return;
    }

    const float *result = mfcc_compute(g_audio_frame);
    if (!result) {
        ESP_LOGE(TAG, "MFCC compute failed");
        mfcc_deinit();
        return;
    }

    const float *mfcc_out = mfcc_output();
    if (!mfcc_out) {
        ESP_LOGE(TAG, "MFCC output is null");
        mfcc_deinit();
        return;
    }

    float *psram_copy = (float *)heap_caps_malloc(MFCC_NUM * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!psram_copy) {
        ESP_LOGE(TAG, "Failed to allocate PSRAM for MFCC");
        mfcc_deinit();
        return;
    }

    memcpy(psram_copy, mfcc_out, MFCC_NUM * sizeof(float));

    ESP_LOGI(TAG, "Dumping MFCC to PSRAM:");
    for (int i = 0; i < MFCC_NUM; ++i) {
        ESP_LOGI(TAG, "MFCC[%d] = %f", i, psram_copy[i]);
    }

    // === LSTM ===
    if (!lstm_init()) {
        ESP_LOGE(TAG, "LSTM init failed");
        heap_caps_free(psram_copy);
        mfcc_deinit();
        return;
    }

    float *lstm_input_buf = (float *)heap_caps_malloc(LSTM_TIMESTEPS * MFCC_NUM * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!lstm_input_buf) {
        ESP_LOGE(TAG, "Failed to allocate LSTM input buffer");
        lstm_deinit();
        heap_caps_free(psram_copy);
        mfcc_deinit();
        return;
    }

    for (int i = 0; i < LSTM_TIMESTEPS; ++i) {
        memcpy(&lstm_input_buf[i * MFCC_NUM], psram_copy, MFCC_NUM * sizeof(float));
    }

    if (!lstm_set_input((const uint8_t *)lstm_input_buf, LSTM_TIMESTEPS * MFCC_NUM * sizeof(float))) {
        ESP_LOGE(TAG, "Failed to set LSTM input");
        heap_caps_free(lstm_input_buf);
        lstm_deinit();
        heap_caps_free(psram_copy);
        mfcc_deinit();
        return;
    }

    if (!lstm_invoke()) {
        ESP_LOGE(TAG, "LSTM inference failed");
        heap_caps_free(lstm_input_buf);
        lstm_deinit();
        heap_caps_free(psram_copy);
        mfcc_deinit();
        return;
    }

    size_t out_len = 0;
    const float *lstm_out = lstm_get_output(&out_len);
    if (!lstm_out || out_len == 0) {
        ESP_LOGE(TAG, "Invalid LSTM output");
        heap_caps_free(lstm_input_buf);
        lstm_deinit();
        heap_caps_free(psram_copy);
        mfcc_deinit();
        return;
    }

    ESP_LOGI(TAG, "LSTM Output:");
    for (size_t i = 0; i < out_len; ++i) {
        ESP_LOGI(TAG, "OUT[%d] = %f", (int)i, lstm_out[i]);
    }

    heap_caps_free(lstm_input_buf);
    lstm_deinit();
    heap_caps_free(psram_copy);
    mfcc_deinit();
    ESP_LOGI(TAG, "=== END MFCC TEST ===");
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "Starting data transfer test...");
    run_mfcc_test();
    ESP_LOGI(TAG, "Data transfer test completed");
}
