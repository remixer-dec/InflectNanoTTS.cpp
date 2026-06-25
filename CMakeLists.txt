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

string(TOLOWER "${CMAKE_SYSTEM_NAME}" INFLECT_TARGET_OS)
if(INFLECT_TARGET_OS STREQUAL "darwin")
    set(INFLECT_TARGET_OS "macos")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" INFLECT_TARGET_ARCH)
if(INFLECT_TARGET_ARCH STREQUAL "amd64")
    set(INFLECT_TARGET_ARCH "x86_64")
elseif(INFLECT_TARGET_ARCH STREQUAL "aarch64")
    set(INFLECT_TARGET_ARCH "arm64")
endif()

set(INFLECT_TARGET_TRIPLE "${INFLECT_TARGET_OS}-${INFLECT_TARGET_ARCH}")

add_subdirectory(ggml)

add_executable(inflect-nano
    src/main.cpp
    src/synthesizer.cpp
    src/griffin_lim_vocoder.cpp
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

set_target_properties(inflect-nano PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${INFLECT_TARGET_TRIPLE}"
)
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(inflect-nano PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/${INFLECT_TARGET_TRIPLE}"
    )
endforeach()

if(INFLECT_LOW_MEMORY)
    target_compile_definitions(inflect-nano PRIVATE INFLECT_LOW_MEMORY)
endif()
