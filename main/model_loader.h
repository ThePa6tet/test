#pragma once

#include <cstdint>
#include <cstddef>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

struct ModelResources {
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    uint8_t *tensor_arena;
    uint8_t *model_data;
    size_t model_size;
};

// Инициализация модели, возврат nullptr в случае ошибки
ModelResources* load_model(uint8_t *model_data, size_t model_size, size_t tensor_arena_size);

// Освобождение ресурсов
void free_model(ModelResources *res);
