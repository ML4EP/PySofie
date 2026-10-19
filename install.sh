#!/usr/bin/env bash
# Install sofie in one of two modes:
#
#   ./install.sh prod [--parsers LIST] [--alpaka --arch ARCH]
#                       installs the package, without the tests. By default it has the ONNX parser
#                       only, and no Python dependencies. The other parsers are the extras of the
#                       package, as a comma-separated LIST:
#                         keras   also the Keras parser (installs numpy, keras, tensorflow)
#                         torch   also the PyTorch parser (installs numpy, torch >= 2.5)
#                         full    all three parsers
#                       for example: ./install.sh prod --parsers keras,torch
#   ./install.sh dev [--alpaka --arch ARCH]
#                       installs what is needed to compile and run the generated code (C++ compiler,
#                       CMake, git, BLAS) and to run the tests (pytest, numpy, keras, tensorflow,
#                       torch) as an editable install, then runs the tests
#
# Both modes take --alpaka --arch ARCH: the generation of portable code is off by default,
# and is for a backend, ARCH: cuda (nvidia) or hip (amd). With it, the package is built for that backend
# (sofie.alpaka_backend), and in dev mode alpaka and sofieBLAS are fetched and the alpaka tests run (for
# the models of the ONNX, Keras and PyTorch parsers).
#
# Environment variables:
#   PYTHON              Python interpreter to install into
#   SOFIE_SOURCE_DIR    SOFIE sources (default: the git submodule SOFIE/, initialized if it is empty)
#   SOFIE_DEPS_DIR      where dev --alpaka puts alpaka and sofieBLAS (default: ./.deps)
#   TORCH_INDEX_URL     where torch is installed from (default: the CPU-only wheels; "default" for PyPI)
#   DRY_RUN=1           only print the commands
set -euo pipefail

usage() {
  echo "usage: $0 dev [--alpaka --arch cuda|hip] | $0 prod [--parsers keras,torch|full] [--alpaka --arch cuda|hip]" >&2
  exit 2
}

mode="${1:-}"
[ "$#" -gt 0 ] && shift
case "$mode" in
  prod | dev) ;;
  *) usage ;;
esac

alpaka=0
alpaka_arch=none
extras=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --alpaka) alpaka=1; shift ;;
    --arch=*) alpaka_arch="${1#--arch=}"; shift ;;
    --arch)
      [ "$#" -ge 2 ] || { echo "--arch needs a value: cuda or hip" >&2; usage; }
      alpaka_arch="$2"; shift 2 ;;
    --parsers=*) parsers="${1#--parsers=}"; shift ;;
    --parsers)
      [ "$#" -ge 2 ] || { echo "--parsers needs a value: keras, torch or full" >&2; usage; }
      parsers="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
  if [ -n "${parsers:-}" ]; then
    [ "$mode" = prod ] || { echo "--parsers is for prod mode (dev installs all of them)" >&2; usage; }
    IFS=, read -r -a listed <<< "$parsers"
    for parser in "${listed[@]}"; do
      case "$parser" in
        onnx) ;;  # the base: always there
        keras | torch | full) extras+=("$parser") ;;
        *) echo "unknown parser: $parser (keras, torch or full)" >&2; usage ;;
      esac
    done
    parsers=""
  fi
done
case "$alpaka:$alpaka_arch" in
  0:none) ;;
  1:cuda | 1:hip) ;;
  1:none) echo "--alpaka needs --arch cuda|hip" >&2; usage ;;
  0:*) echo "--arch is for --alpaka" >&2; usage ;;
  *) echo "unknown arch: $alpaka_arch (cuda or hip)" >&2; usage ;;
esac
alpaka_backend="$alpaka_arch"

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
SOFIE_SOURCE_DIR="${SOFIE_SOURCE_DIR:-$PWD/SOFIE}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  echo "+ $*"
  if [ "$DRY_RUN" != 1 ]; then "$@"; fi
}

have() { command -v "$1" > /dev/null 2>&1; }

# torch from the CPU-only wheels: the default ones bring the CUDA stack, which the parser does not need
install_torch_cpu() {
  if ! "$PYTHON" -c "import torch" > /dev/null 2>&1 && [ "$TORCH_INDEX_URL" != default ]; then
    run "$PYTHON" -m pip install "torch>=2.5" --index-url "$TORCH_INDEX_URL"
  fi
}

# ---- SOFIE sources (needed to build the package in both modes)
if [ ! -f "$SOFIE_SOURCE_DIR/core/CMakeLists.txt" ]; then
  echo "SOFIE sources not found in $SOFIE_SOURCE_DIR"
  if [ "$SOFIE_SOURCE_DIR" = "$PWD/SOFIE" ] && [ -e .git ]; then
    run ./packaging/init_sofie.sh
  else
    echo "error: give the SOFIE sources in SOFIE_SOURCE_DIR (https://github.com/ML4EP/SOFIE)" >&2
    exit 1
  fi
fi
cmake_define="-Ccmake.define.SOFIE_SOURCE_DIR=$SOFIE_SOURCE_DIR"
# the generation of alpaka code is opt-in: the backend is a build option of pip (see packaging/sofie_build.py)
alpaka_option=()
if [ "$alpaka_backend" != none ]; then alpaka_option=("-Calpaka=$alpaka_backend"); fi

# ---- prod mode: the package, with the parsers that were asked for
if [ "$mode" = prod ]; then
  if ! have cmake || ! { have g++ || have c++; }; then
    echo "error: building the package needs a C++20 compiler and CMake (prod mode does not install them)" >&2
    exit 1
  fi
  spec="."
  if [ "${#extras[@]}" -gt 0 ]; then
    spec=".[$(IFS=,; echo "${extras[*]}")]"
    case "$spec" in *torch* | *full*) install_torch_cpu ;; esac
  fi
  run "$PYTHON" -m pip install "$cmake_define" ${alpaka_option[@]+"${alpaka_option[@]}"} "$spec"
  echo
  parsers="onnx"
  for e in "${extras[@]}"; do
    if [ "$e" = full ]; then parsers="onnx keras torch"; else parsers="$parsers $e"; fi
  done
  echo "sofie installed (prod) with the parsers: $parsers. The ONNX parser needs nothing else;"
  echo "add the Keras and PyTorch parsers with ./install.sh prod --parsers keras,torch (or full)."
  echo "To compile the generated code you need a C++20 compiler and a BLAS library."
  if [ "$alpaka_backend" != none ]; then
    echo "alpaka code can be generated, for the $alpaka_backend backend; compile it against alpaka and sofieBLAS yourself."
  else
    echo "The generation of alpaka code is off: add --alpaka --arch cuda (or hip) to enable it."
  fi
  exit 0
fi

# ---- dev mode: system packages
missing=()
have g++ || missing+=(compiler)
have cmake || missing+=(cmake)
have git || missing+=(git)
if have g++ && [ "$(uname)" != Darwin ]; then
  # BLAS is only checked by trying to link it, as the generated code does
  if ! { echo 'int main() {}' | g++ -x c++ - -lopenblas -o /dev/null 2> /dev/null ||
    echo 'int main() {}' | g++ -x c++ - -lblas -o /dev/null 2> /dev/null; }; then
    missing+=(blas)
  fi
fi

if [ "${#missing[@]}" -gt 0 ]; then
  echo "missing system dependencies: ${missing[*]}"
  if have apt-get; then
    run sudo apt-get update
    run sudo apt-get install -y build-essential cmake git libopenblas-dev
  elif have dnf; then
    run sudo dnf install -y gcc-c++ cmake git openblas-devel
  elif have brew; then
    run brew install cmake git openblas
  else
    echo "error: install a C++20 compiler, CMake and a BLAS library (e.g. OpenBLAS) and run this again" >&2
    exit 1
  fi
fi

# ---- dev mode with an alpaka backend: alpaka and sofieBLAS (header-only), at the versions that SOFIE's own tests use
SOFIE_DEPS_DIR="${SOFIE_DEPS_DIR:-$PWD/.deps}"

# fetch_dep <name> <default url> <default tag>
fetch_dep() {
  local name="$1" url="$2" tag="$3" cmake_file="$SOFIE_SOURCE_DIR/tests/CMakeLists.txt" field
  if [ -f "$cmake_file" ]; then
    for field in GIT_REPOSITORY GIT_TAG; do
      value=$(awk -v n="$name" -v f="$field" '
        /FetchContent_Declare/ { getline; gsub(/[ \t]/, ""); cur = $0; next }
        cur == n && $1 == f { print $2; exit }' "$cmake_file")
      if [ -n "$value" ]; then
        if [ "$field" = GIT_REPOSITORY ]; then url="$value"; else tag="$value"; fi
      fi
    done
  fi
  local dir="$SOFIE_DEPS_DIR/$name"
  if [ ! -d "$dir/.git" ]; then
    run mkdir -p "$dir"
    run git -C "$dir" init -q
    run git -C "$dir" remote add origin "$url"
  fi
  run git -C "$dir" fetch -q --depth 1 origin "$tag"
  run git -C "$dir" checkout -q FETCH_HEAD
}

if [ "$alpaka_backend" != none ]; then
  fetch_dep alpaka https://github.com/alpaka-group/alpaka 2fa91a34ed11b2076e474c5507d920e85cf9b79d
  fetch_dep sofieBLAS https://github.com/ML4EP/sofieBLAS dev
  export SOFIE_DEPS_DIR
fi

# ---- dev mode: python packages
install_torch_cpu
run "$PYTHON" -m pip install -e ".[dev]" "$cmake_define" ${alpaka_option[@]+"${alpaka_option[@]}"}

# ---- dev mode: tests (one process per file, so that a crash does not hide the other results)
status=0
while IFS= read -r f; do
  run "$PYTHON" -m pytest -v "$f" || status=1
done < <(find tests -name 'sofie_*.py' | sort)
exit $status
