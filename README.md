# InflectNanoTTS

Small C++/GGML port of the [Inflect-Nano TTS](https://huggingface.co/owensong/Inflect-Nano-v1) pipeline.

This repo contains the runtime, conversion helpers, and parity/debug tooling.

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

With the fallback script:

```bash
./tools/build.sh
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
