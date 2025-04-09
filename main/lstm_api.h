// File: main/lstm_api.h

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool lstm_init();
void lstm_deinit();
bool lstm_set_input(const uint8_t* data, size_t length);
bool lstm_invoke();
const float* lstm_get_output(size_t* out_len);
bool lstm_set_input_f32(const float* data, size_t length);


// 👇 ЭТИ ДВА — нужны для прямого доступа к input/output тензорам
struct TfLiteTensor;
TfLiteTensor* lstm_input();
TfLiteTensor* lstm_output();

#ifdef __cplusplus
}
#endif
