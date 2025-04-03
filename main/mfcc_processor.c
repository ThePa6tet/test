// mfcc_processor.c
#include <math.h>
#include "esp_dsp.h"
#include "dsps_fft2r.h"
#include "dsps_wind.h"
#include "dsps_dct.h"
#include "mfcc_processor.h"

static float window[FRAME_LEN];
static float fft_input[FRAME_LEN * 2];
static float fft_mag[FRAME_LEN / 2];
static float mel_energies[MEL_BANDS];
static float windowed[FRAME_LEN];

void mfcc_preprocess_init(void) {
    dsps_wind_hann_f32(window, FRAME_LEN);
    dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);  // ← Обязательно!
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

void mfcc_preprocess(const float *input_frame, float *mfcc_out) {
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
        mfcc_out[i] = mel_energies[i];
    }
}