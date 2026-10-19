#!/usr/bin/env bash
# Initialize the SOFIE submodule with only core, parsers and cmake, and
# tests/CMakeLists.txt. The benchmark models
# and SOFIE's own tests are not checked out.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
git submodule update --init --depth 1 --filter=blob:none SOFIE
git -C SOFIE sparse-checkout set --no-cone '/CMakeLists.txt' '/LICENSE' '/core/' '/parsers/' '/cmake/' '/tests/CMakeLists.txt'
