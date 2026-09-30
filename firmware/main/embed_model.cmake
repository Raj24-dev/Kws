# Turns the files you drop into the "model" folder into C arrays that get compiled into the firmware.
#   model/<anything>.tflite   the trained microWakeWord model (from Colab Step 13)      -> required for detection
#   model/<anything>.json     the model manifest (threshold, window size, wake word name) -> optional
#   model/<anything>.wav      a 16 kHz 16-bit mono recording of the wake word            -> optional self-test
# The model array is 16-byte aligned (TensorFlow Lite Micro and the ESP-NN SIMD kernels want that).

set(WW_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../model")
file(GLOB WW_TFLITE CONFIGURE_DEPENDS "${WW_MODEL_DIR}/*.tflite")
file(GLOB WW_JSON   CONFIGURE_DEPENDS "${WW_MODEL_DIR}/*.json")
file(GLOB WW_WAV    CONFIGURE_DEPENDS "${WW_MODEL_DIR}/*.wav")

foreach(kind TFLITE JSON WAV)
    list(LENGTH WW_${kind} _n)
    if(_n GREATER 1)
        message(FATAL_ERROR "The model folder contains more than one .${kind} file (${WW_${kind}}). Keep only one.")
    endif()
endforeach()

# Writes "const unsigned char NAME[]" (+ NAME_len) for INPUT, or an empty placeholder if INPUT is empty.
function(ww_bin2c INPUT NAME ADD_NUL OUT_VAR)
    set(_len 0)
    set(_body "0x00")
    if(INPUT)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${INPUT}")
        file(READ "${INPUT}" _hex HEX)
        string(LENGTH "${_hex}" _hexlen)
        math(EXPR _len "${_hexlen} / 2")
        if(ADD_NUL)
            string(APPEND _hex "00")
        endif()
        if(_len GREATER 0)
            string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _body "${_hex}")
        endif()
    endif()
    set(${OUT_VAR} "const unsigned char ${NAME}[] __attribute__((aligned(16))) = {${_body}};\nconst size_t ${NAME}_len = ${_len};\n" PARENT_SCOPE)
endfunction()

ww_bin2c("${WW_TFLITE}" g_ww_model    FALSE _c_model)
ww_bin2c("${WW_JSON}"   g_ww_manifest TRUE  _c_json)
ww_bin2c("${WW_WAV}"    g_ww_selftest FALSE _c_wav)

set(_model_file_name "")
if(WW_TFLITE)
    get_filename_component(_model_file_name "${WW_TFLITE}" NAME)
    message(STATUS "Wake word model: ${_model_file_name}")
else()
    message(WARNING "No .tflite model found in ${WW_MODEL_DIR}. The firmware will run in microphone-test mode.")
endif()

set(WW_GENERATED_SRC "${CMAKE_CURRENT_BINARY_DIR}/ww_model_data.c")
file(WRITE "${WW_GENERATED_SRC}.tmp"
    "// AUTO-GENERATED from the model folder by embed_model.cmake. Do not edit.\n"
    "#include <stddef.h>\n"
    "const char g_ww_model_file[] = \"${_model_file_name}\";\n"
    "${_c_model}${_c_json}${_c_wav}")
configure_file("${WW_GENERATED_SRC}.tmp" "${WW_GENERATED_SRC}" COPYONLY)
