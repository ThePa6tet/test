// File: main/main.cpp

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lstm_api.h"
#include "esp_log.h"
#include "tensorflow/lite/c/common.h"


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

    // Example input: fill with dummy values
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

extern "C" void app_main() {
    xTaskCreate(&run_lstm_task, "lstm_task", 8192, nullptr, 5, nullptr);
}
