import os
import unittest

import numpy as np

import sofie


def parsed_mlp(work_dir, parser, batch_size=2):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch_size, 4)).astype(np.float32)

    if parser == "keras":
        try:
            import keras
        except ImportError:
            raise unittest.SkipTest("keras is not installed")
        model = keras.Sequential(
            [keras.Input(shape=(4,)), keras.layers.Dense(8, activation="relu"), keras.layers.Dense(3)]
        )
        path = os.path.join(work_dir, "mlp.keras")
        model.save(path)
        return sofie.keras.parse(path, batch_size), x, np.asarray(model(x)).ravel()

    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch is not installed")
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).eval()
    expected = model(torch.from_numpy(x)).detach().numpy().ravel()

    if parser == "torch":
        return sofie.pytorch.parse(torch.export.export(model, (torch.from_numpy(x),))), x, expected

    if parser == "onnx":
        try:
            import onnx  # noqa: F401
            import onnxscript  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnx and onnxscript are not installed")
        path = os.path.join(work_dir, "mlp.onnx")
        torch.onnx.export(model, (torch.from_numpy(x),), path, dynamo=True, external_data=False)
        return sofie.onnx.parse(path), x, expected

    raise ValueError(f"unknown parser: {parser}")
