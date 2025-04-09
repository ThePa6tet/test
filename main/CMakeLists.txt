set(SRCS
    main.cpp
    lstm_api.cpp
    mfcc_api.cpp  
    sdcard_manager.cpp
)

set(INCLUDES
    .
)

idf_component_register(SRCS "sdcard_manager.cpp" "lstm_api.cpp" "main.cpp" "${SRCS}"
                       INCLUDE_DIRS "${INCLUDES}")
