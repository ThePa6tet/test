set(SRCS
    main.cpp
    lstm_api.cpp
    mfcc_api.cpp   # ← добавить этот файл!
)

set(INCLUDES
    .
)

idf_component_register(SRCS "${SRCS}"
                       INCLUDE_DIRS "${INCLUDES}")
