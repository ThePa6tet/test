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
#include "model.h"  // модель из header-файла
#include "esp_heap_caps.h"
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include "esp_task_wdt.h"
#include <inttypes.h>


#define TAG "MAIN"  // Определите TAG здесь, до использования в ESP_LOGI
#define MAX_SAMPLES 16000
#define HOP_SIZE 128
#define MAX_FRAMES 99
#define TENSOR_ARENA_SIZE (600 * 1024)

typedef struct {
    float mfcc[MAX_FRAMES][MFCC_NUM];
    char filename[128];
} mfcc_packet_t;

static QueueHandle_t mfcc_queue;
static ModelResources *model = nullptr;



void mfcc_task(void *arg) {
    const char *folders[] = {"d", "u"};
    float sample_buffer[MAX_SAMPLES];

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
            if (!strstr(entry->d_name, ".csv")) continue;

            char path[512];
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

            size_t num_samples = 0;
            if (read_csv_file(path, sample_buffer, MAX_SAMPLES, &num_samples) != ESP_OK) {
                ESP_LOGW(TAG, "Пропуск: %s", path);
                continue;
            }

            mfcc_packet_t packet = {};
            int timestep = 0;
            for (int i = 0; i + FRAME_LEN <= num_samples && timestep < MAX_FRAMES; i += HOP_SIZE) {
                mfcc_preprocess(&sample_buffer[i], packet.mfcc[timestep]);
                timestep++;
            }

            if (timestep < MAX_FRAMES) {
                ESP_LOGW(TAG, "Недостаточно фреймов: %s", path);
                continue;
            }

            strncpy(packet.filename, entry->d_name, sizeof(packet.filename));
            xQueueSend(mfcc_queue, &packet, portMAX_DELAY);
        }

        closedir(dir);
    }

    vTaskDelete(NULL);
}

void infer_task(void *arg) {
    mfcc_packet_t packet;

    while (xQueueReceive(mfcc_queue, &packet, portMAX_DELAY)) {
        if (model->interpreter->input_tensor(0)->type != kTfLiteFloat32) {
            ESP_LOGE(TAG, "Input tensor не float32!");
            continue;
        }
        float *input = model->interpreter->typed_input_tensor<float>(0);
        
        if (!input) {
            ESP_LOGE(TAG, "input tensor == nullptr для %s", packet.filename);
            continue;
        }

        memcpy(input, packet.mfcc, sizeof(packet.mfcc));

        // Проверка output перед инференсом
        TfLiteTensor* output_tensor = model->interpreter->output_tensor(0);
        if (!output_tensor) {
            ESP_LOGE(TAG, "output tensor == nullptr для %s", packet.filename);
            continue;
        }

        if (model->interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke не выполнен для %s", packet.filename);
            continue;
        }

        if (output_tensor->type != kTfLiteFloat32) {
            ESP_LOGE(TAG, "Output tensor имеет неверный тип: %d", output_tensor->type);
            continue;
        }
        

        float result = 0.0f;
        if (output_tensor->type == kTfLiteFloat32) {
            result = output_tensor->data.f[0];
        } else if (output_tensor->type == kTfLiteInt8) {
            int8_t val = output_tensor->data.int8[0];
            result = (val - output_tensor->params.zero_point) * output_tensor->params.scale;
        } else {
            ESP_LOGE(TAG, "Неподдерживаемый тип output: %d", output_tensor->type);
            continue;
        }

        ESP_LOGI("RESULT", "%s → %.3f", packet.filename, result);

        if (result > 0.5f) rgb_led_green();
        else rgb_led_red();

        UBaseType_t items_waiting = uxQueueMessagesWaiting(mfcc_queue);
        ESP_LOGI(TAG, "Очередь MFCC: %u элементов ожидают", (unsigned)items_waiting);

        vTaskDelay(pdMS_TO_TICKS(300));
        rgb_led_off();
    }

    vTaskDelete(NULL);
}



extern "C" void app_main() {
    esp_task_wdt_deinit();  // отключает Task Watchdog полностью
    ESP_LOGI(TAG, "app_main стартует");

    // Инициализация SD-карты
    sdmmc_card_t *card;
    if (!init_sdcard(&card)) {
        ESP_LOGE(TAG, "Ошибка SD-карты");
        return;
    }

    // Копирование модели в PSRAM
    uint8_t *model_data = (uint8_t *)heap_caps_malloc(lstm_mfcc_model_len, MALLOC_CAP_SPIRAM);
    if (!model_data) {
        ESP_LOGE(TAG, "Не удалось выделить память для модели");
        return;
    }
    memcpy(model_data, lstm_mfcc_model, lstm_mfcc_model_len);
    ESP_LOGI(TAG, "Модель загружена в PSRAM (%u байт)", lstm_mfcc_model_len);

    // Инициализация модели
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



    // Остальная инициализация
    mfcc_preprocess_init();
    rgb_led_init();

    // Очередь и задачи
    mfcc_queue = xQueueCreate(4, sizeof(mfcc_packet_t));
    if (!mfcc_queue) {
        ESP_LOGE(TAG, "Ошибка создания очереди");
        return;
    }

    xTaskCreatePinnedToCore(mfcc_task, "mfcc_task", 8192, NULL, 5, NULL, 0);  // CPU 0
    xTaskCreatePinnedToCore(infer_task, "infer_task", 8192, NULL, 5, NULL, 1); // CPU 1
}
