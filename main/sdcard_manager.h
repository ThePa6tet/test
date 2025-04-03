#pragma once

#include "sdmmc_cmd.h"

// Функция инициализации SD-карты
bool init_sdcard(sdmmc_card_t **out_card);

// Функция загрузки файла с SD-карты
uint8_t* load_file_from_sd(const char *path, size_t max_size, size_t *out_size);
esp_err_t read_single_csv_from_d(float *buffer, size_t max_samples, size_t *out_samples);
esp_err_t read_csv_file(const char *file_path, float *buffer, size_t max_samples, size_t *out_samples);