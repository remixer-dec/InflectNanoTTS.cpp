cmake_minimum_required(VERSION 3.16)

project(inflect-nano-tts LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
option(INFLECT_LOW_MEMORY "Enable compile-time low-memory defaults for edge devices" OFF)

add_subdirectory(ggml)

add_executable(inflect-nano
    src/main.cpp
    src/synthesizer.cpp
    src/model_loader.cpp
    src/acoustic_model.cpp
    src/vocoder_model.cpp
    src/text_frontend.cpp
)

target_include_directories(inflect-nano PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

target_link_libraries(inflect-nano PRIVATE ggml)

if(INFLECT_LOW_MEMORY)
    target_compile_definitions(inflect-nano PRIVATE INFLECT_LOW_MEMORY)
endif()
