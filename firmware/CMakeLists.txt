# SIH KWS firmware for ESP32-S3 + INMP441
# Build:  idf.py set-target esp32s3  ->  idf.py menuconfig (Wi-Fi etc.)  ->  idf.py build flash monitor
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(kws_s3)
