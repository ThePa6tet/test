// mfcc_processor.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define FRAME_LEN   256
#define MEL_BANDS   20
#define MFCC_NUM    13
#define PI          3.14159265359
#define SAMPLE_RATE 16000

void mfcc_preprocess_init(void);
void mfcc_preprocess(const float *input_frame, float *mfcc_out);

#ifdef __cplusplus
}
#endif