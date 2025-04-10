idf_component_register(
    SRCS
        main.cpp
        lstm_api.cpp
        mfcc_api.cpp
        sdcard_manager.cpp
    INCLUDE_DIRS
        .
)

# Добавляем флаги компиляции для esp_nn
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    ESP_NN_ENABLE_LOG=1
    LOG_LOCAL_LEVEL=ESP_LOG_DEBUG
)
