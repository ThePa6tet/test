// File: main/mfcc_api.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#define FRAME_LEN   256
#define MEL_BANDS   20
#define MFCC_NUM    13

// Initialize internal buffers and DSP components
bool mfcc_init(void);

// Run MFCC computation on a single audio frame (expects FRAME_LEN samples)
const float *mfcc_compute(const float *audio_frame);

// Get pointer to the latest MFCC output (MFCC_NUM coefficients)
const float *mfcc_output(void);

// Free all allocated resources
void mfcc_deinit(void);

#ifdef __cplusplus
}
#endif
