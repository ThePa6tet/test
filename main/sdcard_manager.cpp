#include "sdcard_manager.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include "esp_heap_caps.h"

#define TAG "SD_UTILS"
#define MOUNT_POINT "/sdcard"

#define PIN_NUM_MISO 13
#define PIN_NUM_MOSI 11
#define PIN_NUM_CLK  12
#define PIN_NUM_CS   GPIO_NUM_5

bool init_sdcard(sdmmc_card_t **out_card) {
    spi_bus_config_t bus_cfg = {
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = PIN_NUM_MISO,
        .sclk_io_num = PIN_NUM_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };

    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus_cfg, SDSPI_DEFAULT_DMA));

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SPI2_HOST;

    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = PIN_NUM_CS;
    slot_config.host_id = static_cast<spi_host_device_t>(host.slot);

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };

    esp_err_t ret = esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot_config, &mount_config, out_card);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Ошибка монтирования SD-карты");
        return false;
    }

    ESP_LOGI(TAG, "SD-карта смонтирована");

    // Выводим список файлов
    DIR* dir = opendir("/sdcard");
    if (dir) {
        struct dirent* ent;
        ESP_LOGI(TAG, "Файлы на SD-карте:");
        while ((ent = readdir(dir)) != NULL) {
            ESP_LOGI(TAG, "  %s", ent->d_name);
        }
        closedir(dir);
    }

    return true;
}

uint8_t* load_file_from_sd(const char *path, size_t max_size, size_t *out_size) {
    struct stat st;
    if (stat(path, &st) != 0) {
        ESP_LOGE(TAG, "Файл не найден: %s", path);
        return nullptr;
    }

    if (st.st_size > max_size) {
        ESP_LOGE(TAG, "Файл слишком большой: %ld байт (макс %u)", st.st_size, max_size);
        return nullptr;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        ESP_LOGE(TAG, "Не удалось открыть файл: %s", path);
        return nullptr;
    }

    uint8_t *buffer = (uint8_t *)heap_caps_malloc(st.st_size, MALLOC_CAP_SPIRAM);
    if (!buffer) {
        ESP_LOGE(TAG, "Ошибка выделения памяти: %ld байт", st.st_size);
        fclose(f);
        return nullptr;
    }

    size_t read = fread(buffer, 1, st.st_size, f);
    fclose(f);

    if (read != st.st_size) {
        ESP_LOGE(TAG, "Ошибка чтения файла");
        free(buffer);
        return nullptr;
    }

    *out_size = read;
    ESP_LOGI(TAG, "Файл успешно загружен (%u байт): %s", read, path);
    return buffer;
}

esp_err_t read_csv_file(const char *file_path, float *buffer, size_t max_samples, size_t *out_samples) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        ESP_LOGE(TAG, "Failed to open file: %s", file_path);
        return ESP_FAIL;
    }

    char line[32];
    size_t index = 0;
    while (fgets(line, sizeof(line), file) && index < max_samples) {
        buffer[index] = strtof(line, NULL);
        index++;
    }

    fclose(file);

    if (out_samples) {
        *out_samples = index;
    }

    ESP_LOGI(TAG, "CSV file read successfully: %s, samples: %d", file_path, (int)index);
    return ESP_OK;
}

// Упрощённый вариант: читаем один файл напрямую
esp_err_t read_single_csv_from_d(float *buffer, size_t max_samples, size_t *out_samples) {
    DIR *dir = opendir(MOUNT_POINT "/d");
    if (!dir) {
        ESP_LOGE(TAG, "Failed to open /sdcard/d");
        return ESP_FAIL;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".CSV") || strstr(entry->d_name, ".csv")) {
            char full_path[300];
            snprintf(full_path, sizeof(full_path), MOUNT_POINT "/d/%s", entry->d_name);
            closedir(dir);
            ESP_LOGI(TAG, "Reading file: %s", full_path);
            return read_csv_file(full_path, buffer, max_samples, out_samples);
        }
    }

    closedir(dir);
    ESP_LOGW(TAG, "No CSV files found in /sdcard/d");
    return ESP_FAIL;
}