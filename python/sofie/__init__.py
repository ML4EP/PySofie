"""Python interface to SOFIE, tool that translates trained ML models to self-contained C++ code.

``sofie`` exposes the standalone SOFIE C++ library (https://github.com/ML4EP/SOFIE) through a 
pybind11 extension module (``sofie._sofie``): ``RModel``, ``ETensorType``,
the ``ROperator_*`` classes, ... Templated operators are bound for float and are spelled
``ROperator_Gemm["float"](...)``.

On top of it, the parsers ``sofie.keras.parse``, ``sofie.pytorch.parse`` and ``sofie.onnx.parse``
translate trained Keras models (.keras files), PyTorch models exported with ``torch.export`` (.pt2
files or ``ExportedProgram`` objects) and ONNX models (.onnx files) into a SOFIE ``RModel``, 
from which standalone C++ inference code is generated.

The heavyweight dependencies (keras/tensorflow, torch) are imported lazily inside the Parse()
functions, so importing this package is cheap.
"""

from . import _sofie
from ._sofie import *

class _templated_operator:
    """Mimics the C++ template instantiation syntax: ``Op["float"]`` (or ``Op("float")``)."""

    def __init__(self, name, instances):
        self._name = name
        self._instances = instances

    def __getitem__(self, dtype):
        try:
            return self._instances[dtype]
        except KeyError:
            raise RuntimeError(
                f"SOFIE - Unsupported - Operator {self._name} is not available for type {dtype}"
                f" (available: {', '.join(self._instances)})"
            ) from None

    __call__ = __getitem__


_templated = [
    "Gemm",
    "Conv",
    "Relu",
    "Selu",
    "Sigmoid",
    "Tanh",
    "LeakyRelu",
    "Elu",
    "Identity",
    "Pool",
    "BatchNormalization",
    "LayerNormalization",
    "RNN",
    "GRU",
    "LSTM",
    "Erf",
    "HardSwish",
    "HardSigmoid",
    "Where",
    "Clip",
    "Expand",
    "GroupNorm",
    "SDPA",
    "Tile",
    "Pad",
    "CumSum",
    "InstanceNormalization",
    "ConvTranspose",
]
for _name in _templated:
    globals()["ROperator_" + _name] = _templated_operator(_name, {"float": getattr(_sofie, f"ROperator_{_name}_float")})
del _name

__version__ = "0.1.0"

# The parsers are subpackages: sofie.onnx, sofie.keras and sofie.pytorch. They (and the packages they
# use: numpy, keras, torch) are only loaded when they are used.
_parsers = ("onnx", "keras", "pytorch")


def __getattr__(name):
    if name in _parsers:
        import importlib

        try:
            return importlib.import_module("." + name, __name__)
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[0] == __name__:
                raise
            extra = {"keras": "keras", "pytorch": "torch"}.get(name, "full")
            raise ImportError(
                f"sofie.{name} needs the package '{e.name}', which is not installed. Install it with the extra "
                f'of sofie that has it: pip install "sofie[{extra}]" (the base install has the ONNX parser only)'
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
