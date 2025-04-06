// File: main/lstm_api.cpp

#include "lstm_api.h"
#include "model.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include <inttypes.h>

constexpr int kTensorArenaSize = 512 * 1024;
static const char *TAG = "LSTM_API";

static uint8_t *tensor_arena = nullptr;
static uint8_t *model_copy = nullptr;
static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;

static tflite::MicroMutableOpResolver<14> resolver;

bool lstm_init() {
    if (interpreter != nullptr) {
        ESP_LOGW(TAG, "Interpreter already initialized");
        return true;
    }

    ESP_LOGI(TAG, "Allocating tensor arena (%d bytes)...", kTensorArenaSize);
    tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    ESP_LOGI(TAG, "Allocating model copy (%zu bytes)...", lstm_mfcc_model_len);
    model_copy   = (uint8_t *)heap_caps_malloc(lstm_mfcc_model_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!tensor_arena || !model_copy) {
        ESP_LOGE(TAG, "Failed to allocate tensor_arena or model_copy");
        return false;
    }

    memcpy(model_copy, lstm_mfcc_model, lstm_mfcc_model_len);
    model = tflite::GetModel(model_copy);
    ESP_LOGI(TAG, "Model loaded to address: %p", model);

    if (!model) {
        ESP_LOGE(TAG, "Model is nullptr after GetModel");
        return false;
    }

    ESP_LOGI(TAG, "Model schema version: %u (expected %u)",
             static_cast<unsigned int>(model->version()), static_cast<unsigned int>(TFLITE_SCHEMA_VERSION));

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch");
        return false;
    }

    ESP_LOGI(TAG, "Registering operators...");
    resolver.AddFullyConnected(); resolver.AddRelu(); resolver.AddSoftmax();
    resolver.AddShape(); resolver.AddStridedSlice(); resolver.AddTranspose();
    resolver.AddUnpack(); resolver.AddPack(); resolver.AddFill();
    resolver.AddAdd(); resolver.AddSplit(); resolver.AddLogistic();
    resolver.AddMul(); resolver.AddTanh();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    ESP_LOGI(TAG, "Allocating tensors...");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return false;
    }

    ESP_LOGI(TAG, "LSTM model initialization successful.");
    return true;
}

TfLiteTensor *lstm_input() {
    return interpreter ? interpreter->input(0) : nullptr;
}

TfLiteTensor *lstm_output() {
    return interpreter ? interpreter->output(0) : nullptr;
}

bool lstm_invoke() {
    if (!interpreter) return false;
    return interpreter->Invoke() == kTfLiteOk;
}

void lstm_deinit() {
    if (tensor_arena) {
        ESP_LOGI(TAG, "Freeing tensor_arena...");
        heap_caps_free(tensor_arena);
        tensor_arena = NULL;
    }
    if (model_copy) {
        ESP_LOGI(TAG, "Freeing model_copy...");
        heap_caps_free(model_copy);
        model_copy = nullptr;
    }
    ESP_LOGI(TAG, "Deinitializing interpreter and model pointers");
    interpreter = nullptr;
    model = nullptr;
}

