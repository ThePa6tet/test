// File: main/mfcc_api.cpp

#include "mfcc_api.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <cmath>
#include <cstring>

#define TAG "MFCC_API"

static float *g_mfcc_output = nullptr;

bool mfcc_init(void) {
    ESP_LOGI(TAG, "Allocating buffers...");
    g_mfcc_output = (float *)heap_caps_malloc(sizeof(float) * MFCC_NUM, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!g_mfcc_output) {
        ESP_LOGE(TAG, "MFCC output buffer allocation failed");
        return false;
    }

    // Optionally: zero initialize
    memset(g_mfcc_output, 0, sizeof(float) * MFCC_NUM);
    ESP_LOGI(TAG, "MFCC buffers initialized");
    return true;
}

const float *mfcc_compute(const float *audio_frame) {
    if (!g_mfcc_output || !audio_frame) return nullptr;

    ESP_LOGI(TAG, "MFCC_API: Computing MFCCs...");
    for (int i = 0; i < MFCC_NUM; ++i) {
        g_mfcc_output[i] = (float)(i * 10) + sinf(i);  // mock
    }

    return g_mfcc_output;
}

const float *mfcc_output(void) {
    return g_mfcc_output;
}

void mfcc_deinit(void) {
    ESP_LOGI(TAG, "MFCC_API: Deinitializing MFCC buffers...");
    if (g_mfcc_output) {
        heap_caps_free(g_mfcc_output);
        g_mfcc_output = nullptr;
    }
}
