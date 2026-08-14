# sofie-extra-parsers

Keras and PyTorch model parsers for [TMVA
SOFIE](https://root.cern/manual/tmva/#sofie), distributed as a standalone
Python package.

These parsers were part of [ROOT](https://root.cern) until they were removed
from the ROOT repository (see commit `64a153fc95a`, *"[tmva][sofie] Remove
Keras and PyTorch parses from ROOT repository"*), when the SOFIE distribution
within ROOT was slimmed down to the core ONNX-based functionality. This
package continues to provide them on top of ROOT.

The parsers translate a trained Keras model (`.keras` file) or a TorchScript
model (`.pt` file) into a `TMVA::Experimental::SOFIE::RModel`, from which
SOFIE generates standalone C++ inference code.

## Requirements

- **ROOT** with PyROOT and TMVA/SOFIE enabled. ROOT is not distributed on
  PyPI, so it cannot be installed by pip; it has to come from your
  environment (system package manager, conda-forge, CVMFS, nix, a source
  build, ...). The parsers work with ROOT 6.40 and later; note that with
  ROOT 6.40, dilated Conv2D models produce wrong inference results because
  of a bug in SOFIE's Conv operator that is only fixed in later versions
  (root-project/root commit d42a7a7780c).
- **keras**/**tensorflow** for parsing Keras models, **torch** for parsing
  TorchScript models. These are imported lazily, so you only need the ones
  you actually use.

## Installation

```sh
pip install .
```

Optional extras: `pip install .[keras]`, `pip install .[torch]`, or
`pip install .[test]` for everything needed to run the test suite.

## Usage

Parsing a Keras model:

```python
from sofie_extra_parsers import PyKeras

rmodel = PyKeras.Parse("model.keras", batch_size=1)
rmodel.Generate()
rmodel.OutputGenerated("model.hxx")
```

Parsing a TorchScript model (the input tensor shapes have to be given
explicitly):

```python
from sofie_extra_parsers import PyTorch

rmodel = PyTorch.Parse("model.pt", [[1, 4]])
rmodel.Generate()
rmodel.OutputGenerated("model.hxx")
```

The generated header can then be compiled and used for inference, e.g. by
just-in-time compiling it with ROOT:

```python
import ROOT

ROOT.gInterpreter.Declare('#include "model.hxx"')
session = ROOT.TMVA_SOFIE_model.Session("model.dat")
output = session.infer(input_numpy_array)
```

The `tutorials/` directory contains the two end-to-end examples
(`TMVA_SOFIE_Keras.py` and `TMVA_SOFIE_PyTorch.py`) that were removed from
ROOT together with the parsers, adapted to the new API.

## Development

A [nix](https://nixos.org) dev shell with ROOT, Python 3.13 and all test
dependencies (tensorflow, torch, onnx, pytest) is provided:

```sh
nix develop
python -m pytest -v
```

Note that the flake pins Python 3.13 because tensorflow is not available for
Python 3.14, the current default in nixpkgs. ROOT is built against the same
Python version so that PyROOT is importable; since that is not the default
configuration of the nixpkgs `root` package, it is not in the binary cache
and will be built from source the first time.

The test suite also runs in CI on AlmaLinux 10, where ROOT is available as a
system package from EPEL.

## License

LGPL-2.1-or-later, like ROOT, from which this code originates.
