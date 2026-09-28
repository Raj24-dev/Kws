set(srcs
    "main.c"
    "audio_input.c"
    "wake_word.cpp"
    "streamer.c"
    "wifi.c"
    "status_led.c"
    "wav_selftest.c"
)

# Embed the model files (skipped during ESP-IDF's early "requirements" pass)
if(NOT CMAKE_BUILD_EARLY_EXPANSION)
    include("${CMAKE_CURRENT_LIST_DIR}/embed_model.cmake")
    list(APPEND srcs "${WW_GENERATED_SRC}")
endif()

# No REQUIRES list: the "main" component then depends on every component in the project,
# including the ones downloaded from idf_component.yml.
idf_component_register(SRCS ${srcs}
                       INCLUDE_DIRS ".")
