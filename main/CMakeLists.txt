idf_component_register(
    SRCS 
        "main.cpp"
        "sdcard_manager.cpp"
        "model_loader.cpp"
        "rgb_led.c"
        "mfcc_processor.c"
    INCLUDE_DIRS "."  
    REQUIRES
        esp-dsp
        fatfs
        vfs
        sdmmc
        led_strip
        esp-tflite-micro
)
