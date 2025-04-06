// File: main/main.cpp

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lstm_api.h"
#include "mfcc_api.h"
#include "esp_log.h"
#include "tensorflow/lite/c/common.h"
#include <cmath>
#include "esp_heap_trace.h"

#define NUM_RECORDS 256
static heap_trace_record_t trace_record[NUM_RECORDS];


static const char *TAG = "MAIN";

extern "C" void run_lstm_task(void *) {
    if (!lstm_init()) {
        ESP_LOGE(TAG, "Failed to initialize LSTM model");
        vTaskDelete(nullptr);
        return;
    }

    TfLiteTensor *input = lstm_input();
    if (!input) {
        ESP_LOGE(TAG, "Failed to get input tensor");
        vTaskDelete(nullptr);
        return;
    }

    for (int i = 0; i < input->bytes / sizeof(float); ++i) {
        reinterpret_cast<float *>(input->data.raw)[i] = 0.5f;
    }

    if (!lstm_invoke()) {
        ESP_LOGE(TAG, "LSTM inference failed");
        vTaskDelete(nullptr);
        return;
    }

    TfLiteTensor *output = lstm_output();
    if (!output) {
        ESP_LOGE(TAG, "Failed to get output tensor");
        vTaskDelete(nullptr);
        return;
    }

    float *out_data = reinterpret_cast<float *>(output->data.raw);
    size_t out_len = output->bytes / sizeof(float);

    for (size_t i = 0; i < out_len; ++i) {
        ESP_LOGI(TAG, "Output[%zu] = %f", i, out_data[i]);
    }

    lstm_deinit();
    vTaskDelete(nullptr);
}

extern "C" void run_mfcc_task(void *) {
    ESP_LOGI(TAG, "Free heap check: %d", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    ESP_ERROR_CHECK_WITHOUT_ABORT(heap_caps_check_integrity_all(true)); // если хочешь
    if (!mfcc_init()) {
        ESP_LOGE(TAG, "Failed to initialize MFCC");
        vTaskDelete(nullptr);
        return;
    }

    float dummy_audio[FRAME_LEN] = {0};
    for (int i = 0; i < FRAME_LEN; ++i) {
        dummy_audio[i] = sinf(2 * 3.14159f * 440.0f * i / 16000.0f);

    }

    const float *mfcc = mfcc_compute(dummy_audio);
    if (!mfcc) {
        ESP_LOGE(TAG, "MFCC compute failed");
        vTaskDelete(nullptr);
        return;
    }

    for (int i = 0; i < MFCC_NUM; ++i) {
        ESP_LOGI(TAG, "MFCC[%d] = %f", i, mfcc[i]);
    }

    mfcc_deinit();
    heap_trace_stop();
    heap_trace_dump();

    vTaskDelete(nullptr);
}

extern "C" void app_main() {
    ESP_ERROR_CHECK(heap_trace_init_standalone(trace_record, NUM_RECORDS));
    ESP_ERROR_CHECK(heap_trace_start(HEAP_TRACE_LEAKS));

    xTaskCreatePinnedToCore(&run_lstm_task, "lstm_task", 8192, nullptr, 5, nullptr, 0);
    xTaskCreatePinnedToCore(&run_mfcc_task, "mfcc_task", 8192, nullptr, 5, nullptr, 1);
}
