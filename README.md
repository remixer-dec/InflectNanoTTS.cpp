# InflectSanoTTS.cpp

Small C++/GGML runtime for
[Inflect-Nano TTS v1](https://huggingface.co/owensong/Inflect-Nano-v1),
[Inflect-Nano TTS v2](https://huggingface.co/owensong/Inflect-Nano-v2/), and
[Inflect-Micro TTS v2](https://huggingface.co/owensong/Inflect-Micro-v2/)
pipelines, plus the [sanoTTS](https://github.com/Ampixa/sanoTTS)
Piperlite architecture.

This repo contains the runtime, conversion helpers, and parity/debug tooling. CPU inference is supported, including the low-memory ESP32-S3 path.
Pre-quantized weights are available here:
[V1-Nano](https://huggingface.co/remixerdec/Inflect-Nano-v1-GGUF),
[V2-Nano](https://huggingface.co/remixerdec/Inflect-Nano-v2-GGUF), and
[V2-Micro](https://huggingface.co/remixerdec/Inflect-Micro-v2-GGUF).
Sano Piperlite voice quants and audio samples are available in the
[SanoTTS-GGUF release](https://huggingface.co/remixerdec/SanoTTS-GGUF), which
includes Amy, Amy-small, HFC, Kristin, Vietnamese, and Indonesian voices.

## Layout

- `src/`: runtime and CLI
- `tools/`: fallback build and parity helpers, conversion scripts
- `tools/convert.py`: v1 PyTorch checkpoint to GGUF conversion
- `tools/convert_v2.py`: v2 checkpoint conversion and GGUF quantization
- `tools/compile_cmudict.py`: compile `cmudict.rep` into `cmudict.bin`
- `tools/compile-v2-lexicon.py`: compile the v2 eSpeak lexicon into `lexicon.bin`
- `tools/convert_sano.py`: convert a canonical sanoTTS Piperlite GGUF
- `tools/compile-sano-lexicon.py`: compile a Piper phoneme lexicon for Sano
- `ggml/`: vendored GGML submodule

## Build

With CMake:

```bash
cmake -S . -B build
cmake --build build -j
```

Both CMake and `tools/build.sh` place the CLI binary under `build/<os>-<arch>/inflect-sano`.

With the fallback script:

```bash
./tools/build.sh
```

### Low-memory build

For edge devices, compile with `INFLECT_LOW_MEMORY`. This enables flash/file-backed CMU lookup, defers vocoder loading until after acoustic inference, releases acoustic memory before vocoding, and uses smaller vocoder chunks. Low-memory build currently stays under 9MB of RAM during inference on Linux and under 7MB on ESP32's PSRAM. On an ESP32, Inflect-Nano v1 with the Griffin-Lim vocoder takes around 6 seconds from cold start to a spoken single word. Use `INFLECT_MEM_TRACE=1` env. variable to trace memory usage.

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
./build/<os>-<arch>/inflect-sano \
  -a /path/to/inflect_acoustic.gguf \
  -v /path/to/inflect_vocoder.gguf \
  -d /path/to/cmudict.bin \
  -t "Hello, this is a test." \
  -o output.wav
```

## Inflect Nano v2

V2 uses one unified `inflect-v2` GGUF plus a flash-backed `IVL2` lexicon.

Nano-v2 and Micro-v2 use the same GGUF architecture identifier. Active channel
dimensions are read from GGUF metadata; the converter reads them from each
release's `config.json`. Runtime inference never reads a PyTorch checkpoint or
configuration JSON.

```bash
./build/<os>-<arch>/inflect-sano \
  --model-family v2 \
  --v2-model /path/to/inflect_nano_v2_f32.gguf \
  --v2-lexicon /path/to/lexicon.bin \
  --speed 1.0 \
  --variation 0.667 \
  --seed 7 \
  --decoder-chunk-frames 32 \
  -t "A small voice can still have something meaningful to say." \
  -o output.wav
```

`--variation` controls latent sampling noise: `0` disables it and `1` uses the
full learned variance. `--seed` makes synthesis deterministic.

`--token-file tokens.txt` bypasses text processing with a whitespace- or
comma-separated list of unblanked token IDs. The runtime inserts blank token
`0` between them, which is useful for testing exact pronunciations or frontend
output.

## Sano Piperlite

InflectSano supports the Piperlite graph from
[sanoTTS](https://github.com/Ampixa/sanoTTS). This support is for Piperlite;
the separate Sano Heart/Nano architecture is not part of this runtime.

Conversion and lexicon generation run on the host. The converter needs Python,
NumPy, and the `gguf` package. Lexicon generation also needs the host eSpeak
and phonemizer dependencies used by sanoTTS.

Convert the canonical Sano Piperlite GGUF and compile its runtime lexicon:

```bash
python tools/convert_sano.py \
  --input /path/to/canonical-sano-piperlite.gguf \
  --config /path/to/manifest.json \
  --phoneme-config /path/to/piper-phoneme-config.json \
  --output /path/to/sano_piperlite.gguf \
  --quantize q4_0_e

python tools/compile-sano-lexicon.py \
  --phoneme-config /path/to/piper-phoneme-config.json \
  --cmudict /path/to/cmudict.bin \
  --format snl2 \
  --output /path/to/sano_lexicon.snl
```

Use the existing build and run instructions above with these Sano-specific
options:

- `--model-family sano`: select the Sano Piperlite runtime
- `--sano-model /path/to/sano_piperlite.gguf`: load the converted model
- `--sano-lexicon /path/to/sano_lexicon.snl`: load the compiled lexicon
- `--speaking-rate 1.0`: set the Sano speaking rate

### Griffin-Lim backend

The neural vocoder is the default quality path. For lower memory and faster experiments, add:

```bash
+  --vocoder-backend griffin_lim
```

This synthesizes waveform audio from the acoustic mel output with Griffin-Lim and gives a robotic vibe to it. The vocoder model is not loaded, so `-v` is not used. You can also set `INFLECT_VOCODER_BACKEND=griffin_lim`.

Griffin-Lim applies only to v1. V1 exposes an 80-bin mel magnitude estimate
before vocoding; v2 is a VITS model whose expanded latent is consumed directly
by its learned waveform decoder. The v2 latent is not a mel or STFT magnitude,
so passing it to Griffin-Lim would not be a valid alternate decoder.
