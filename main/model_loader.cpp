// Стандартные и POSIX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

// ESP-IDF
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

// SD-карта
#include "sdmmc_cmd.h"
#include "esp_vfs_fat.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"

// LED
#include "led_strip.h"
#include "rgb_led.h"  // твой модуль

// MFCC
#include "mfcc_processor.h"

// Модель и инференс
#include "model_loader.h"

// Файл с CSV
#include "sdcard_manager.h"  // если read_csv_file там


#define TAG "MODEL_LOADER"

#define MAX_SAMPLES    16000
#define HOP_SIZE       128
float sample_buffer[MAX_SAMPLES];


ModelResources* load_model(uint8_t *model_data, size_t model_size, size_t tensor_arena_size) {
    auto *res = new ModelResources();

    res->model_data = model_data;
    res->model_size = model_size;

    // Парсинг модели
    res->model = tflite::GetModel(model_data);
    if (res->model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Неверная версия модели");
        delete res;
        return nullptr;
    }

    // Выделение памяти под tensor_arena
    res->tensor_arena = (uint8_t *)heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM);
    if (!res->tensor_arena) {
        ESP_LOGE(TAG, "Не удалось выделить tensor_arena (%d байт)", (int)tensor_arena_size);
        delete res;
        return nullptr;
    }

    // Создание резолвера
    static tflite::MicroMutableOpResolver<14> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddTranspose();
    resolver.AddUnpack();
    resolver.AddPack();
    resolver.AddFill();
    resolver.AddAdd();
    resolver.AddSplit();
    resolver.AddLogistic();
    resolver.AddMul();
    resolver.AddTanh();
    // добавь другие операции при необходимости

    // Создание интерпретатора
    res->interpreter = new tflite::MicroInterpreter(res->model, resolver, res->tensor_arena, tensor_arena_size);
    
    TfLiteStatus status = res->interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors не удался");
        free(res->tensor_arena);
        delete res->interpreter;
        delete res;
        return nullptr;
    }

    ESP_LOGI(TAG, "Интерпретатор готов. Тензоров: %lu", res->model->subgraphs()->Get(0)->tensors()->size());

    return res;
}

void free_model(ModelResources *res) {
    if (!res) return;
    if (res->interpreter) delete res->interpreter;
    if (res->tensor_arena) free(res->tensor_arena);
    // model_data очищается отдельно, если нужно
    delete res;
}
