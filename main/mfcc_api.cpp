// File: main/mfcc_api.cpp

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_dsp.h"
#include "dsps_fft2r.h"
#include "dsps_wind.h"
#include "dsps_dct.h"
#include "mfcc_api.h"
#include "esp_heap_caps.h"

#define PI          3.14159265359f
#define SAMPLE_RATE 16000

static const char *TAG = "MFCC_API";

static float *window;
static float *fft_input;
static float *fft_mag;
static float *mel_energies;
static float *windowed;
static float *output_mfcc;

#ifdef __cplusplus
extern "C" {
#endif

bool mfcc_init(void) {
    if (window != NULL) {
        ESP_LOGW(TAG, "mfcc_init already called. Call mfcc_deinit first.");
        return false;
    }
    

    ESP_LOGI(TAG, "Allocating buffers...");
    window = fft_input = fft_mag = mel_energies = windowed = output_mfcc = NULL;
    
    window        = (float*)heap_caps_malloc(FRAME_LEN * sizeof(float), MALLOC_CAP_INTERNAL);
    fft_input     = (float*)heap_caps_malloc(FRAME_LEN * 2 * sizeof(float), MALLOC_CAP_INTERNAL);
    fft_mag       = (float*)heap_caps_malloc((FRAME_LEN / 2) * sizeof(float), MALLOC_CAP_INTERNAL);
    mel_energies  = (float*)heap_caps_malloc(MEL_BANDS * sizeof(float), MALLOC_CAP_INTERNAL);
    windowed      = (float*)heap_caps_malloc(FRAME_LEN * sizeof(float), MALLOC_CAP_INTERNAL);
    output_mfcc   = (float*)heap_caps_malloc(MFCC_NUM * sizeof(float), MALLOC_CAP_INTERNAL);

    if (!window || !fft_input || !fft_mag || !mel_energies || !windowed || !output_mfcc) {
        ESP_LOGE(TAG, "MFCC buffer allocation failed");
        return false;
    }

    dsps_wind_hann_f32(window, FRAME_LEN);
    dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    ESP_LOGI(TAG, "MFCC buffers initialized");
    return true;
}

static void compute_magnitude(const float *fft, float *mag, int len) {
    for (int i = 0; i < len; i++) {
        float real = fft[i * 2 + 0];
        float imag = fft[i * 2 + 1];
        mag[i] = fabsf(real) + fabsf(imag);
    }
}

static void mel_filterbank(const float *spectrum, float *mel_out, int fft_bins, int mel_bands, int sample_rate) {
    float f_min = 300.0f;
    float f_max = sample_rate / 2.0f;
    float mel_min = 2595.0f * log10f(1.0f + f_min / 700.0f);
    float mel_max = 2595.0f * log10f(1.0f + f_max / 700.0f);

    float mel_points[MEL_BANDS + 2];
    for (int i = 0; i < MEL_BANDS + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * i / (MEL_BANDS + 1);
        float hz = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
        mel_points[i] = (fft_bins + 1) * hz / (sample_rate / 2.0f);
    }

    for (int m = 0; m < mel_bands; m++) {
        mel_out[m] = 0;
        int left = (int)mel_points[m];
        int center = (int)mel_points[m + 1];
        int right = (int)mel_points[m + 2];

        for (int k = left; k < center; k++) {
            if (k < fft_bins) {
                float w = (k - mel_points[m]) / (mel_points[m + 1] - mel_points[m]);
                mel_out[m] += spectrum[k] * w;
            }
        }
        for (int k = center; k < right; k++) {
            if (k < fft_bins) {
                float w = (mel_points[m + 2] - k) / (mel_points[m + 2] - mel_points[m + 1]);
                mel_out[m] += spectrum[k] * w;
            }
        }

        if (mel_out[m] < 1e-6f) mel_out[m] = 1e-6f;
    }
}

const float* mfcc_compute(const float *input_frame) {
    ESP_LOGI(TAG, "Computing MFCCs...");

    dsps_mul_f32((float *)input_frame, window, windowed, FRAME_LEN, 1, 1, 1);
    for (int i = 0; i < FRAME_LEN; i++) {
        fft_input[i * 2 + 0] = windowed[i];
        fft_input[i * 2 + 1] = 0;
    }

    dsps_fft2r_fc32(fft_input, FRAME_LEN);
    dsps_bit_rev_fc32(fft_input, FRAME_LEN);
    dsps_cplx2reC_fc32(fft_input, FRAME_LEN);

    compute_magnitude(fft_input, fft_mag, FRAME_LEN / 2);
    mel_filterbank(fft_mag, mel_energies, FRAME_LEN / 2, MEL_BANDS, SAMPLE_RATE);
    dsps_dct_f32(mel_energies, MEL_BANDS);

    for (int i = 0; i < MFCC_NUM; i++) {
        output_mfcc[i] = mel_energies[i];
    }

    return output_mfcc;
}


const float* mfcc_output(void) {
    return output_mfcc;
}

void mfcc_deinit(void) {
    ESP_LOGI(TAG, "Deinitializing MFCC buffers...");

    if (window) free(window);
    if (fft_input) free(fft_input);
    if (fft_mag) free(fft_mag);
    if (mel_energies) free(mel_energies);
    if (windowed) free(windowed);
    if (output_mfcc) free(output_mfcc);

    window = fft_input = fft_mag = mel_energies = windowed = output_mfcc = nullptr;
}

#ifdef __cplusplus
}
#endif
