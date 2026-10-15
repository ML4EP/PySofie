### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### This macro provides a simple example for the parsing of Keras .keras file
### into RModel object and further generating the .hxx header files for inference.
###
### \macro_code
### \macro_output
### \author Sanjiban Sengupta and Lorenzo Moneta


import os
import contextlib
import warnings

import numpy as np
import sofie

os.makedirs("generated", exist_ok=True)
from tensorflow.keras.layers import Activation, Dense, Input, Softmax
from tensorflow.keras.models import Model


@contextlib.contextmanager
def expect_warning(category, message):
    """Silence a known third-party warning and raise if it stops firing.

    Notifies us to drop the workaround once the upstream library is fixed.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    seen = False
    for w in caught:
        if issubclass(w.category, category) and message in str(w.message):
            seen = True
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if not seen:
        raise RuntimeError(
            f"Expected {category.__name__} containing {message!r} was not "
            "emitted. This tutorial's workaround can probably be removed."
        )


# -----------------------------------------------------------------------------
# Step 1: Create and train a simple Keras model (via embedded Python)
# -----------------------------------------------------------------------------

input = Input(shape=(4,), batch_size=2)
x = Dense(32)(input)
x = Activation("relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(2)(x)
output = Softmax()(x)
model = Model(inputs=input, outputs=output)

random_generator = np.random.RandomState(0)
x_train = random_generator.rand(4, 4)
y_train = random_generator.rand(4, 2)

model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=3, batch_size=2)

# Keras' internal ``np.array(x)`` (TensorFlow backend) does not yet implement
# the numpy 2.0 ``__array__(copy=...)`` signature, so saving the model emits a
# DeprecationWarning that we cannot fix from user code.
if tuple(int(p) for p in np.__version__.split(".")[:2]) >= (2, 0):
    ctx = expect_warning(DeprecationWarning, "__array__ implementation doesn't accept a copy keyword")
else:
    ctx = contextlib.nullcontext()

with ctx:
    model.save("generated/KerasModel.keras")

model.summary()

# -----------------------------------------------------------------------------
# Step 2: Use SOFIE to parse the Keras model
# -----------------------------------------------------------------------------

model = sofie.keras.parse("generated/KerasModel.keras")

# Generate the inference code and write it to KerasModel.hxx
model.generate()
model.output_generated("generated/KerasModel.hxx")

# print generated code
print("\n**************************************************")
print("           Generated code")
print("**************************************************\n")
model.print_generated()
print("**************************************************\n\n\n")

# -----------------------------------------------------------------------------
# Step 3: Use the generated code
# -----------------------------------------------------------------------------

# KerasModel.hxx is a standalone C++ header: include it in your C++ code, link
# with BLAS, and run inference with
#
#   #include "generated/KerasModel.hxx"
#   SOFIE_KerasModel::Session session("KerasModel.dat");
#   std::vector<float> y = session.infer(x);   // x: float const* of shape (2, 4)
