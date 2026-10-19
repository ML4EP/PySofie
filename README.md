# PySofie

Python bindings for [SOFIE](https://github.com/ML4EP/SOFIE).

## How it works

The SOFIE C++ library (`core/`) is built together with a
[pybind11](https://pybind11.readthedocs.io) extension module
(`sofie._sofie`, see `src/`) that is installed with
the package. `import sofie` gives the Python interface
([`python/sofie/__init__.py`](python/sofie/__init__.py): `RModel`,
`ETensorType`, `ROperator_*`, ...); templated operators are bound for `float`
and are spelled `ROperator_Gemm["float"](...)`.
Operators are passed to `RModel.add_operator`, which takes ownership of them.

The bindings in `src/` are split by what they bind: `types.cpp` (tensor types,
options, `Dim`), `rmodel.cpp` (`RModel`), `roperator.cpp` (the `ROperator` base
class) and one file per operator in `src/operators/` (for example `gemm.cpp`).
`bindings.cpp` only calls them, in order. To bind another SOFIE operator, add
`src/operators/<name>.cpp` with a `bind_<name>` function, declare it in
`src/bindings.hpp`, call it from `bindings.cpp` and list the file in
`CMakeLists.txt`.

## Installation

The base install has the ONNX parser only and no Python dependencies. The Keras
and PyTorch parsers are extras of the package that install the frameworks they
need:

```sh
pip install .             # the ONNX parser (no dependencies)
pip install ".[keras]"    # + the Keras parser: numpy, keras, tensorflow
pip install ".[torch]"    # + the PyTorch parser: numpy, torch >= 2.5
pip install ".[full]"     # all three parsers
```

Extras can be combined (`".[keras,torch]"`). The Python code of all three parsers
is always installed; the extras decide which frameworks come with it. Using a
parser whose framework is missing raises an error that names the extra to
install. `import sofie`, `RModel`, the operators and `generate` need nothing else;
numpy is only imported when tensor data is added to a model (which the Keras
and PyTorch parsers do; the ONNX parser does not).

`install.sh` does the same and also installs the PyTorch CPU wheels (the default
ones bring the CUDA stack; `TORCH_INDEX_URL=default` to use PyPI's), and has a
`dev` mode for developing:

```sh
./install.sh prod                                       # the ONNX parser
./install.sh prod --parsers keras,torch                 # + the Keras and PyTorch parsers (or: --parsers full)
./install.sh prod --parsers full --alpaka --arch cuda   # + alpaka code generation for nvidia
./install.sh dev                                        # includes tests validating the generated code
./install.sh dev --alpaka --arch cuda                   # + the generation of alpaka code, and its tests
```

The tests and tutorials are not installed in prod mode. Building the package
needs a C++20 compiler and CMake.

**alpaka support is off by default, and is per backend**: nvidia (CUDA) or amd
(HIP). The generation of portable code,
`rmodel.generate_gpu_alpaka`, is only in a build for a backend. Which one a build
is for is `sofie.alpaka_backend` (`"cuda"`, `"hip"` or `None`); in a build
without one, the method raises an error that says how to get it, and
`sofie.with_alpaka` is `False`. The parsers are chosen by the extras and the
backend by a build option, which pip passes with `-C`:

```sh
pip install "sofie[full]" -C alpaka=cuda      # all parsers, and alpaka for nvidia gpu
pip install "sofie[torch]" -C alpaka=hip                  # torch and onnx parsers only, and alpaka for amd gpu
./install.sh prod --parsers full --alpaka --arch cuda    # the same, with the script
```

The generator is the same for every backend (the generated `Session` is templated
on the accelerator): the backend is chosen when the generated code is compiled,
with `nvcc` for CUDA and `hipcc` for HIP.

**dev** installation allows compiling the generated code and running the corresponding tests:
- system packages, through apt, dnf or brew, if missing: a C++ compiler, CMake,
  git and a BLAS library (OpenBLAS; the generated code calls `sgemm_`/`sgemv_`);
- the Python packages of the tests (`pip install -e ".[dev]"`: pytest, numpy,
  keras, tensorflow, torch, and onnx and onnxscript, which `torch.onnx.export` needs
  to make the ONNX models; torch from the CPU-only wheels unless
  `TORCH_INDEX_URL=default`);
- with `--alpaka --arch cuda` (or `hip`) only: the build for that backend, and
  [alpaka](https://github.com/alpaka-group/alpaka) and
  [sofieBLAS](https://github.com/ML4EP/sofieBLAS), which the heterogeneous
  code includes. They are header-only and are fetched into `.deps/` at the
  versions that SOFIE's own tests use. The alpaka tests run only then, for the
  models of all three parsers (ONNX, Keras, PyTorch).

Running the alpaka code itself needs a GPU backend and its toolchain (e.g. the
CUDA toolkit).

SOFIE ([ML4EP/SOFIE](https://github.com/ML4EP/SOFIE)) is a git submodule of
this repository. Clone with
`git clone --recurse-submodules` (or run `git submodule update --init` afterwards;
`install.sh` does it if the directory is empty). To build against another
checkout of SOFIE, set `SOFIE_SOURCE_DIR` (or pass
`-C cmake.define.SOFIE_SOURCE_DIR=/path/to/SOFIE` to pip). Use `PYTHON=/path/to/python` to install into a
particular environment, `SOFIE_DEPS_DIR` to move `.deps/`, and `DRY_RUN=1` to
only print the commands. Without the
script: `pip install ".[full]"` (prod) or `pip install -e ".[dev]"`.

## Usage

Parsing a Keras model:

```python
import sofie

rmodel = sofie.keras.parse("model.keras", batch_size=1)
rmodel.generate()
rmodel.output_generated("model.hxx")
```

Parsing a PyTorch model:

```python
import torch
import sofie

torch.export.save(torch.export.export(model.eval(), (torch.rand(1, 4),)), "model.pt2")
rmodel = sofie.pytorch.parse("model.pt2")   # also accepts an ExportedProgram or an nn.Module
rmodel.generate()
rmodel.output_generated("model.hxx")
```

Parsing an ONNX model, with SOFIE's own ONNX parser. It reads the ONNX format
itself, so it needs neither the `onnx` package nor protobuf, nor any framework:

```python
import sofie

sofie.onnx.check_model("model.onnx")   # prints the operators that SOFIE does not support, returns a bool
rmodel = sofie.onnx.parse("model.onnx")
rmodel.generate()
rmodel.output_generated("model.hxx")
```

The tool only emits the C++ header (and the weight file next to it). The
header is standalone: include it, link against BLAS and use its `Session`
class. It does not include any header of SOFIE, so nothing of SOFIE is needed to
compile it.

For heterogeneous inference, `rmodel.generate_gpu_alpaka(options, batch_size)`
generates [alpaka](https://github.com/alpaka-group/alpaka) code instead of the
CPU code.
Retrieve it with `output_generated` or `return_generated` as above. You
compile and link it yourself, against alpaka,
[sofieBLAS](https://github.com/ML4EP/sofieBLAS) and the toolchain of your GPU
backend (`./install.sh dev --alpaka --arch cuda` fetches alpaka and sofieBLAS for the tests).

The alpaka tests run only in a build for a backend (`--alpaka --arch cuda`). They check
the emitted code and that it compiles against alpaka and sofieBLAS
(`tests/<parser>/alpaka/`), for an MLP parsed by each of the ONNX, Keras and
PyTorch parsers.

The `tutorials/` directory contains two end-to-end examples
(`TMVA_SOFIE_Keras.py` and `TMVA_SOFIE_PyTorch.py`).

## Development

`./install.sh dev` sets everything up and runs the tests. To run them again:

```sh
python -m pytest -v tests/torch/cpu/sofie_torch_parser.py     # or any file in tests/
```

The editable install rebuilds the C++ extension when it changes.

The tests are organised by parser and by target:

```
tests/
  onnx/  torch/  keras/    one directory per parser
    cpu/                   the generated CPU code
    alpaka/                the generated alpaka code: the emitted code and that it compiles
      cuda/                the alpaka code run on a CUDA GPU
  common/                  helpers shared by the tests
```

**Validation of generated code** (`tests/generated_code_runner.py`, test
code only). A test parses a model, emits the header, and then:

1. writes a small CMake project around it: a `main` that reads the inputs from
   files, calls `Session::infer`, and writes the output to a file;
2. builds it with CMake (which finds the compiler, BLAS and, for alpaka, the
   CUDA toolkit; the CPU code is built with the address and
   undefined-behaviour sanitizers, `SOFIE_TEST_SANITIZE=0` turns them off);
3. runs the executable in a separate process, on the same random input that the
   test gave to the original model;
4. compares, in Python, the output of that executable with the output of the
   model in its own framework (PyTorch, Keras, or PyTorch for an ONNX file),
   with `numpy.testing.assert_allclose`.

## License

[LGPL-2.1](https://github.com/ML4EP/PySofie/blob/main/LICENSE)
