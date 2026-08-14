"""Keras and PyTorch model parsers for TMVA SOFIE.

This package provides the Keras and PyTorch parsers that were part of ROOT
until they were removed from the ROOT repository. They translate trained Keras
models (.keras files) and TorchScript models (.pt files) into a
TMVA::Experimental::SOFIE::RModel, from which standalone C++ inference code can
be generated.

The heavyweight dependencies (ROOT, keras/tensorflow, torch) are imported
lazily inside the Parse() functions, so importing this package is cheap.
"""

from .keras.parser import PyKeras
from .pytorch.parser import PyTorch

__version__ = "0.1.0"

__all__ = ["PyKeras", "PyTorch"]
