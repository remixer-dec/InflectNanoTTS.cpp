# InflectNanoTTS.cpp

Small C++/GGML port of the [Inflect-Nano TTS](https://huggingface.co/owensong/Inflect-Nano-v1) pipeline.

This repo contains the runtime, conversion helpers, and parity/debug tooling. Currently only cpu inference is supported. 
Pre-quantized weights are available [here](https://huggingface.co/remixerdec/Inflect-Nano-v1-GGUF).

## Layout

- `src/`: runtime and CLI
- `tools/`: fallback build and parity helpers, conversion scripts
- `convert.py`: PyTorch checkpoint to GGUF conversion
- `compile_cmudict.py`: compile `cmudict.rep` into `cmudict.bin`
- `ggml/`: vendored GGML submodule

## Build

With CMake:

```bash
cmake -S . -B build
cmake --build build -j
```

Both CMake and `tools/build.sh` place the CLI binary under `build/<os>-<arch>/inflect-nano`.

With the fallback script:

```bash
./tools/build.sh
```

### Low-memory build

For edge devices, compile with `INFLECT_LOW_MEMORY`. This enables flash/file-backed CMU lookup, defers vocoder loading until after acoustic inference, releases acoustic memory before vocoding, and uses smaller vocoder chunks. Low-memory build currently stays under 9MB of RAM during inference on Linux and under 7MB on ESP32's PSRAM. Use `INFLECT_MEM_TRACE=1` env. variable to trace memory usage.

```bash
cmake -S . -B build-lowmem -DINFLECT_LOW_MEMORY=ON
cmake --build build-lowmem -j

# or
INFLECT_LOW_MEMORY=1 BUILD_DIR=build/lowmem ./tools/build.sh
```

## GGML patches

Local GGML changes live in `patches/ggml/` so the nested `ggml` submodule can be reset and patched reproducibly.

From this directory, apply them with:

```bash
(cd ggml && git apply ../patches/ggml/*.patch)
```

To refresh the patch after editing `ggml`:

```bash
git -C ggml diff --binary > patches/ggml/0001-esp32-low-memory-support.patch
```

## Run

The CLI expects explicit asset paths:

```bash
./build/<os>-<arch>/inflect-nano \
  -a /path/to/inflect_acoustic.gguf \
  -v /path/to/inflect_vocoder.gguf \
  -d /path/to/cmudict.bin \
  -t "Hello, this is a test." \
  -o output.wav
```

### Griffin-Lim backend

The neural vocoder is the default quality path. For lower memory and faster experiments, add:

```bash
+  --vocoder-backend griffin_lim
```

This synthesizes waveform audio from the acoustic mel output with Griffin-Lim and gives a robotic vibe to it. The vocoder model is not loaded, so `-v` is not used. You can also set `INFLECT_VOCODER_BACKEND=griffin_lim`.
