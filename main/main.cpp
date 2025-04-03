#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_err.h"
#include "sdmmc_cmd.h"
#include "esp_vfs_fat.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "rgb_led.h"
#include "mfcc_processor.h"
#include "model_loader.h"
#include "sdcard_manager.h"
#include "model.h"
#include "esp_heap_caps.h"
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include "esp_task_wdt.h"
#include <inttypes.h>

#define TAG "MAIN"
#define MAX_SAMPLES 16000
#define HOP_SIZE 128
#define MAX_FRAMES 99
#define TENSOR_ARENA_SIZE (600 * 1024)

typedef struct {
    float* mfcc_ptr;
    char filename[128];
} mfcc_packet_ptr_t;

static QueueHandle_t mfcc_queue;
static ModelResources *model = nullptr;

void mfcc_task(void *arg) {
    const char *folders[] = {"d", "u"};
    float sample_buffer[MAX_SAMPLES];

    ESP_LOGI(TAG, "Задача mfcc_task стартует");

    for (int s = 0; s < 2; ++s) {
        char dir_path[128];
        snprintf(dir_path, sizeof(dir_path), "/sdcard/%s", folders[s]);
        DIR *dir = opendir(dir_path);
        if (!dir) {
            ESP_LOGW(TAG, "Не открыть папку: %s", dir_path);
            continue;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (!strstr(entry->d_name, ".CSV")) continue;

            char path[512];
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

            size_t num_samples = 0;
            if (read_csv_file(path, sample_buffer, MAX_SAMPLES, &num_samples) != ESP_OK) {
                ESP_LOGW(TAG, "Пропуск: %s", path);
                continue;
            }

            float* mfcc_data = (float*)heap_caps_malloc(sizeof(float) * MAX_FRAMES * MFCC_NUM, MALLOC_CAP_SPIRAM);
            if (!mfcc_data) {
                ESP_LOGE(TAG, "Не удалось выделить память под MFCC");
                continue;
            }

            int timestep = 0;
            for (int i = 0; i + FRAME_LEN <= num_samples && timestep < MAX_FRAMES; i += HOP_SIZE) {
                mfcc_preprocess(&sample_buffer[i], &mfcc_data[timestep * MFCC_NUM]);
                timestep++;
            }

            if (timestep < MAX_FRAMES) {
                ESP_LOGW(TAG, "Недостаточно фреймов: %s", path);
                heap_caps_free(mfcc_data);
                continue;
            }

            mfcc_packet_ptr_t packet = {};
            packet.mfcc_ptr = mfcc_data;
            strncpy(packet.filename, entry->d_name, sizeof(packet.filename));

            ESP_LOGI(TAG, "Обработка файла: %s", path);
            xQueueSend(mfcc_queue, &packet, portMAX_DELAY);
            vTaskDelay(pdMS_TO_TICKS(50));
        }

        closedir(dir);
    }

    vTaskDelete(NULL);
}

void infer_task(void *arg) {
    ESP_LOGI(TAG, "Задача infer_task стартует");

    mfcc_packet_ptr_t packet;
    int total = 0, correct = 0;

    while (xQueueReceive(mfcc_queue, &packet, portMAX_DELAY)) {
        float *input = model->interpreter->typed_input_tensor<float>(0);
        if (!input || !packet.mfcc_ptr) {
            ESP_LOGE(TAG, "input или mfcc_ptr == nullptr для %s", packet.filename);
            if (packet.mfcc_ptr) heap_caps_free(packet.mfcc_ptr);
            continue;
        }

        memcpy(input, packet.mfcc_ptr, sizeof(float) * MAX_FRAMES * MFCC_NUM);

        TfLiteTensor* output_tensor = model->interpreter->output_tensor(0);
        if (!output_tensor) {
            ESP_LOGE(TAG, "output tensor == nullptr для %s", packet.filename);
            heap_caps_free(packet.mfcc_ptr);
            continue;
        }

        if (model->interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke не выполнен для %s", packet.filename);
            heap_caps_free(packet.mfcc_ptr);
            continue;
        }

        float result = output_tensor->data.f[0];

        int label = (strchr(packet.filename, 'D') != nullptr) ? 1 : 0;
        int predicted = (result > 0.5f) ? 1 : 0;
        if (label == predicted) correct++;
        total++;

        float accuracy = 100.0f * correct / total;
        ESP_LOGI("RESULT", "%s → %.3f (%d vs %d)", packet.filename, result, predicted, label);
        ESP_LOGI(TAG, "Точность: %.2f%% (%d из %d)", accuracy, correct, total);

        heap_caps_free(packet.mfcc_ptr);
        vTaskDelay(pdMS_TO_TICKS(300));
    }

    vTaskDelete(NULL);
}

extern "C" void app_main() {
    esp_task_wdt_deinit();
    ESP_LOGI(TAG, "app_main стартует");

    sdmmc_card_t *card;
    if (!init_sdcard(&card)) {
        ESP_LOGE(TAG, "Ошибка SD-карты");
        return;
    }

    uint8_t *model_data = (uint8_t *)heap_caps_malloc(lstm_mfcc_model_len, MALLOC_CAP_SPIRAM);
    if (!model_data) {
        ESP_LOGE(TAG, "Не удалось выделить память для модели");
        return;
    }
    memcpy(model_data, lstm_mfcc_model, lstm_mfcc_model_len);
    ESP_LOGI(TAG, "Модель загружена в PSRAM (%u байт)", lstm_mfcc_model_len);

    model = load_model(model_data, lstm_mfcc_model_len, TENSOR_ARENA_SIZE);
    if (!model) {
        free(model_data);
        ESP_LOGE(TAG, "Не удалось инициализировать модель");
        return;
    }

    ESP_LOGI(TAG, "Интерпретатор готов. Входных тензоров: %zu", static_cast<size_t>(model->interpreter->inputs().size()));

    TfLiteTensor* input_tensor = model->interpreter->input_tensor(0);
    TfLiteTensor* output_tensor = model->interpreter->output_tensor(0);

    if (input_tensor) {
        ESP_LOGI(TAG, "Input tensor dims: %d x %d", input_tensor->dims->data[0], input_tensor->dims->data[1]);
    } else {
        ESP_LOGW(TAG, "Input tensor == nullptr");
    }

    if (output_tensor) {
        ESP_LOGI(TAG, "Output tensor dims: %d", output_tensor->dims->data[0]);
    } else {
        ESP_LOGW(TAG, "Output tensor == nullptr");
    }

    mfcc_preprocess_init();
    rgb_led_init();

    mfcc_queue = xQueueCreate(4, sizeof(mfcc_packet_ptr_t));
    if (!mfcc_queue) {
        ESP_LOGE(TAG, "Ошибка создания очереди");
        return;
    }

    xTaskCreatePinnedToCore(mfcc_task, "mfcc_task", 8192, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(infer_task, "infer_task", 8192, NULL, 5, NULL, 1);
}
