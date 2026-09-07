cmake_minimum_required(VERSION 3.16)

project(inflect-nano-tts LANGUAGES C CXX)
include(CTest)

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
option(INFLECT_PROFILE_ACOUSTIC_OPS
       "Enable profiling-only acoustic operator instrumentation"
       OFF)
option(INFLECT_V2_ESPEAK_FALLBACK
       "Enable CPU-only eSpeak-ng fallback for V2 words missing from IVL2"
       OFF)

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

if(INFLECT_LOW_MEMORY)
    add_compile_definitions(INFLECT_LOW_MEMORY)
endif()

add_subdirectory(ggml)

set(INFLECT_SOURCES
    src/synthesizer.cpp
    src/griffin_lim_vocoder.cpp
    src/model_loader.cpp
    src/acoustic_model.cpp
    src/vocoder_model.cpp
    src/text_frontend.cpp
    src/sano_frontend.cpp
    src/sano_piper_model.cpp
    src/sano_piper_decoder.cpp
    src/v2_symbols.cpp
    src/v2_frontend.cpp
    src/v2_model.cpp
)
add_executable(inflect-nano src/main.cpp ${INFLECT_SOURCES})

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

if(INFLECT_V2_ESPEAK_FALLBACK)
    target_compile_definitions(inflect-nano PRIVATE INFLECT_V2_ESPEAK_FALLBACK)
endif()

if(INFLECT_PROFILE_ACOUSTIC_OPS)
    target_compile_definitions(
        inflect-nano PRIVATE INFLECT_PROFILE_ACOUSTIC_OPS=1)
endif()

if(BUILD_TESTING)
    add_executable(test-sano-pipeline tests/test_sano_pipeline.cpp ${INFLECT_SOURCES})
    target_include_directories(test-sano-pipeline PRIVATE src ggml/include)
    target_compile_definitions(test-sano-pipeline PRIVATE INFLECT_LOW_MEMORY)
    target_link_libraries(test-sano-pipeline PRIVATE ggml)
    add_test(NAME sano-pipeline COMMAND test-sano-pipeline)
    add_executable(test-v2-frontend
        tests/test_v2_frontend.cpp
        src/v2_frontend.cpp
        src/v2_symbols.cpp
    )
    target_include_directories(test-v2-frontend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME v2-frontend COMMAND test-v2-frontend)

    add_executable(test-vocoder-quant-math
        tests/test_vocoder_quant_math.cpp
    )
    target_include_directories(test-vocoder-quant-math PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(
        NAME vocoder-quant-math
        COMMAND test-vocoder-quant-math
    )

    add_executable(test-vocoder-rounding-exhaustive
        tests/test_vocoder_rounding_exhaustive.cpp
    )
    target_include_directories(
        test-vocoder-rounding-exhaustive PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(
        NAME vocoder-rounding-exhaustive
        COMMAND test-vocoder-rounding-exhaustive
    )

    add_executable(test-sano-frontend
        tests/test_sano_frontend.cpp
        src/sano_frontend.cpp
        src/v2_frontend.cpp
        src/v2_symbols.cpp
    )
    target_include_directories(test-sano-frontend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME sano-frontend COMMAND test-sano-frontend)

    add_executable(test-sano-rounding
        tests/test_sano_rounding.cpp
    )
    target_include_directories(test-sano-rounding PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME sano-rounding COMMAND test-sano-rounding)
endif()
