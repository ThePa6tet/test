extern "C" {
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
    #include "sdcard_manager.h"
    }
    #include "model_loader.h"
    
    #include <stdio.h>
    #include <string.h>
    #include <dirent.h>
    
    #define TAG "MAIN"
    #define MODEL_PATH "/sdcard/LSTM_M~1.TFL"
    #define MODEL_MAX_SIZE (800 * 1024)
    #define TENSOR_ARENA_SIZE (350 * 1024)
    
    #define MAX_SAMPLES 16000
    #define HOP_SIZE 128
    #define MAX_FRAMES 78
    
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
            float *input = model->interpreter->typed_input_tensor<float>(0);
            memcpy(input, packet.mfcc, sizeof(packet.mfcc));
    
            if (model->interpreter->Invoke() != kTfLiteOk) {
                ESP_LOGE(TAG, "Инференс не выполнен для %s", packet.filename);
                continue;
            }
    
            float result = model->interpreter->typed_output_tensor<float>(0)[0];
            ESP_LOGI("RESULT", "%s → %.3f", packet.filename, result);
    
            if (result > 0.5f) rgb_led_green();
            else rgb_led_red();
    
            vTaskDelay(pdMS_TO_TICKS(300));
            rgb_led_off();
        }
    
        vTaskDelete(NULL);
    }
    
    extern "C" void app_main() {
        ESP_LOGI(TAG, "app_main стартует");
    
        sdmmc_card_t *card;
        if (!init_sdcard(&card)) {
            ESP_LOGE(TAG, "Ошибка SD-карты");
            return;
        }
    
        size_t model_size;
        uint8_t *model_data = load_file_from_sd(MODEL_PATH, MODEL_MAX_SIZE, &model_size);
        if (!model_data) return;
    
        model = load_model(model_data, model_size, TENSOR_ARENA_SIZE);
        if (!model) {
            free(model_data);
            return;
        }
    
        mfcc_preprocess_init();
        rgb_led_init();
    
        mfcc_queue = xQueueCreate(4, sizeof(mfcc_packet_t));
        if (!mfcc_queue) {
            ESP_LOGE(TAG, "Ошибка создания очереди");
            return;
        }
    
        xTaskCreatePinnedToCore(mfcc_task, "mfcc_task", 8192, NULL, 5, NULL, 0);  // CPU 0
        xTaskCreatePinnedToCore(infer_task, "infer_task", 8192, NULL, 5, NULL, 1); // CPU 1
    }
    