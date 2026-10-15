import os
import shutil
import unittest

import numpy as np
import torch
from generated_code_runner import compile_model, generated_dir
from operator_cases import cases

import sofie

work_dir = generated_dir("pytorch_operator_models")

atol = {"sdpa": 1e-5, "var": 1e-5}


class sofie_torch_operators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

    def test_operators_and_models(self):
        torch.manual_seed(0)
        for name, module, args in cases():
            with self.subTest(name):
                module = module.eval()
                ep = torch.export.export(module, args)
                unsupported = [o["op"] for o in sofie.pytorch.list_operators(ep) if not o["supported"]]
                self.assertEqual(unsupported, [])

                session = compile_model(sofie.pytorch.parse(ep), f"{work_dir}/{name}.hxx")
                expected = module(*args).detach().numpy()
                result = session.infer(*[a.numpy() for a in args])
                self.assertEqual(result.size, expected.size)
                np.testing.assert_allclose(result, expected.flatten(), atol=atol.get(name, 1e-4), rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
