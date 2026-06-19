#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
os_name="$(uname -s)"
arch_name="$(uname -m)"

normalize_os() {
  case "$1" in
    Linux) echo "linux" ;;
    Darwin) echo "macos" ;;
    *) echo "$1" | tr '[:upper:]' '[:lower:]' ;;
  esac
}

normalize_arch() {
  case "$1" in
    x86_64|amd64) echo "x86_64" ;;
    arm64|aarch64) echo "arm64" ;;
    *) echo "$1" | tr '[:upper:]' '[:lower:]' ;;
  esac
}

TARGET_TRIPLE="$(normalize_os "${os_name}")-$(normalize_arch "${arch_name}")"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/${TARGET_TRIPLE}}"
OBJ_DIR="${BUILD_DIR}/obj"
BIN="${BUILD_DIR}/inflect-nano"

CC="${CC:-cc}"
CXX="${CXX:-c++}"

COMMON_FLAGS=(
  -O2
  -DNDEBUG
  -DGGML_USE_CPU
  -DGGML_SCHED_MAX_COPIES=4
  -DGGML_VERSION=\"0.15.1\"
  -DGGML_COMMIT=\"vendored\"
  -I"${ROOT_DIR}"
  -I"${ROOT_DIR}/ggml/include"
  -I"${ROOT_DIR}/ggml/src"
  -I"${ROOT_DIR}/ggml/src/ggml-cpu"
)

case "${os_name}" in
  Linux*) COMMON_FLAGS+=(-D_GNU_SOURCE -D_XOPEN_SOURCE=600) ;;
  Darwin*) COMMON_FLAGS+=(-D_DARWIN_C_SOURCE) ;;
esac

C_FLAGS=(-std=c11 "${COMMON_FLAGS[@]}")
CXX_FLAGS=(-std=c++17 "${COMMON_FLAGS[@]}")
LD_FLAGS=(-pthread)

if [[ "${os_name}" == "Linux" ]]; then
  LD_FLAGS+=(-ldl)
fi

C_SOURCES=(
  ggml/src/ggml.c
  ggml/src/ggml-alloc.c
  ggml/src/ggml-quants.c
  ggml/src/ggml-cpu/ggml-cpu.c
  ggml/src/ggml-cpu/quants.c
)

CXX_SOURCES=(
  src/main.cpp
  src/synthesizer.cpp
  src/model_loader.cpp
  src/acoustic_model.cpp
  src/vocoder_model.cpp
  src/text_frontend.cpp
  ggml/src/ggml.cpp
  ggml/src/ggml-backend.cpp
  ggml/src/ggml-backend-meta.cpp
  ggml/src/ggml-backend-reg.cpp
  ggml/src/ggml-backend-dl.cpp
  ggml/src/ggml-opt.cpp
  ggml/src/ggml-threading.cpp
  ggml/src/gguf.cpp
  ggml/src/ggml-cpu/ggml-cpu.cpp
  ggml/src/ggml-cpu/repack.cpp
  ggml/src/ggml-cpu/hbm.cpp
  ggml/src/ggml-cpu/traits.cpp
  ggml/src/ggml-cpu/binary-ops.cpp
  ggml/src/ggml-cpu/unary-ops.cpp
  ggml/src/ggml-cpu/vec.cpp
  ggml/src/ggml-cpu/ops.cpp
)

case "${arch_name}" in
  x86_64|amd64)
    C_SOURCES+=(ggml/src/ggml-cpu/arch/x86/quants.c)
    CXX_SOURCES+=(
      ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
      ggml/src/ggml-cpu/arch/x86/repack.cpp
      ggml/src/ggml-cpu/amx/amx.cpp
      ggml/src/ggml-cpu/amx/mmq.cpp
    )
    ;;
  arm64|aarch64)
    C_SOURCES+=(ggml/src/ggml-cpu/arch/arm/quants.c)
    CXX_SOURCES+=(
      ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp
      ggml/src/ggml-cpu/arch/arm/repack.cpp
    )
    ;;
esac

mkdir -p "${OBJ_DIR}"

objects=()

compile_c() {
  local src="$1"
  local obj="${OBJ_DIR}/${src//\//_}.o"
  "${CC}" "${C_FLAGS[@]}" -c "${ROOT_DIR}/${src}" -o "${obj}"
  objects+=("${obj}")
}

compile_cxx() {
  local src="$1"
  local obj="${OBJ_DIR}/${src//\//_}.o"
  "${CXX}" "${CXX_FLAGS[@]}" -c "${ROOT_DIR}/${src}" -o "${obj}"
  objects+=("${obj}")
}

for src in "${C_SOURCES[@]}"; do
  compile_c "${src}"
done

for src in "${CXX_SOURCES[@]}"; do
  compile_cxx "${src}"
done

"${CXX}" "${objects[@]}" "${LD_FLAGS[@]}" -o "${BIN}"

echo "${BIN}"
