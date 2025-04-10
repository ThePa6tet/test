#define LOG_LOCAL_LEVEL ESP_LOG_DEBUG    // до esp_log.h
#define ESP_NN_ENABLE_LOG 1              // до esp_nn.h, если подключаешь напрямую

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "mfcc_api.h"
#include "lstm_api.h"
#include "sdcard_manager.h"
#include <cmath>
#include <cstring>
#include <dirent.h>
#include "esp_task_wdt.h"


#define TAG "MAIN"
#define SAMPLE_SIZE 1024
#define FRAME_LEN   256
#define MFCC_NUM    13
#define LSTM_TIMESTEPS 99
#define LSTM_INPUT_SIZE (LSTM_TIMESTEPS * MFCC_NUM)
#define ESPNN_MATH


static float g_audio_frame[FRAME_LEN];

void generate_sine_wave(float *buffer, size_t len, float freq, float rate) {
    for (size_t i = 0; i < len; ++i) {
        buffer[i] = sinf(2.0f * M_PI * freq * i / rate);
    }
}

int run_inference_on_file(const char *full_path, float expected_label) {
    float *csv_data = (float *)heap_caps_malloc(sizeof(float) * LSTM_INPUT_SIZE, MALLOC_CAP_SPIRAM);
    if (!csv_data) return 0;

    size_t samples_read = 0;
    if (read_csv_file(full_path, csv_data, LSTM_INPUT_SIZE, &samples_read) != ESP_OK || samples_read != LSTM_INPUT_SIZE) {
        heap_caps_free(csv_data);
        return 0;
    }

    // ⬇️ Теперь используем float-передачу
    if (!lstm_set_input_f32(csv_data, LSTM_INPUT_SIZE)) {
        ESP_LOGE(TAG, "Failed to set LSTM input");
        heap_caps_free(csv_data);
        return 0;
    }

    if (!lstm_invoke()) {
        ESP_LOGE(TAG, "LSTM invoke failed");
        heap_caps_free(csv_data);
        return 0;
    }

    size_t out_len = 0;
    const float *lstm_out = lstm_get_output(&out_len);
    heap_caps_free(csv_data);

    if (!lstm_out || out_len == 0) return 0;

    float prediction = lstm_out[0];
    int pred_label = roundf(prediction);
    int expected = roundf(expected_label);
    ESP_LOGI(TAG, "Predicted: %.4f -> %d, Expected: %d", prediction, pred_label, expected);
    return pred_label == expected ? 1 : 0;
}

int run_benchmark_on_folder(const char *subfolder, float expected_label, int *out_total) {
    char path[256];
    snprintf(path, sizeof(path), "/sdcard/%s", subfolder);
    DIR *dir = opendir(path);
    if (!dir) {
        ESP_LOGE(TAG, "Failed to open directory: %s", path);
        return 0;
    }

    struct dirent *entry;
    int correct = 0;
    int total = 0;
    char full_path[512];

    while ((entry = readdir(dir)) != NULL && total < 100) {
        //if (strstr(entry->d_name, ".csv") || strstr(entry->d_name, ".CSV")) {
            if (entry->d_type == DT_REG) {
            snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
            ESP_LOGI(TAG, "Testing file: %s", full_path);
            correct += run_inference_on_file(full_path, expected_label);
            total++;
        }
    }
    closedir(dir);
    *out_total = total;
    return correct;
}

void run_test_from_sdcard() {
    sdmmc_card_t *card;
    if (!init_sdcard(&card)) {
        ESP_LOGE(TAG, "SD init failed");
        return;
    }

    if (!lstm_init()) {
        ESP_LOGE(TAG, "LSTM init failed");
        return;
    }

    int total_d = 0, correct_d = run_benchmark_on_folder("D", 1.0f, &total_d);
    int total_u = 0, correct_u = run_benchmark_on_folder("U", 0.0f, &total_u);

    int total = total_d + total_u;
    int correct = correct_d + correct_u;

    float acc_d = (total_d > 0) ? (100.0f * correct_d / total_d) : 0.0f;
    float acc_u = (total_u > 0) ? (100.0f * correct_u / total_u) : 0.0f;
    float acc_total = (total > 0) ? (100.0f * correct / total) : 0.0f;

    ESP_LOGI(TAG, "Accuracy on /d/: %.2f%% (%d/%d)", acc_d, correct_d, total_d);
    ESP_LOGI(TAG, "Accuracy on /u/: %.2f%% (%d/%d)", acc_u, correct_u, total_u);
    ESP_LOGI(TAG, "Total Accuracy: %.2f%% (%d/%d)", acc_total, correct, total);

    lstm_deinit();
}

extern "C" void app_main() {
    esp_task_wdt_deinit();
    ESP_LOGI(TAG, "Starting data transfer test...");
    run_test_from_sdcard();
    ESP_LOGI(TAG, "Data transfer test completed");
}
