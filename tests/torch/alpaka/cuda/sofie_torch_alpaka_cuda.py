import concurrent.futures
import os
import unittest

import numpy as np
import torch
from alpaka_cases import cuda_cases
from generated_code_runner import compile_model_cuda, fresh_generated_dir
from operator_cases import cases
from supported import torch_alpaka_cuda

import sofie


class sofie_torch_alpaka_cuda(cuda_cases, unittest.TestCase):
    parser = "torch"

    @classmethod
    def setUpClass(cls):
        cls.work_dir = fresh_generated_dir("torch_alpaka_cuda")
        torch.manual_seed(0)
        cls.cases = {}
        for name, module, args in cases(torch_alpaka_cuda):
            cls.cases[name] = (module.eval(), args)

        parsed = {}
        for name, (module, args) in cls.cases.items():
            try:
                parsed[name] = sofie.pytorch.parse(torch.export.export(module, args))
            except Exception as e:
                parsed[name] = e

        def build(name):
            if isinstance(parsed[name], Exception):
                return parsed[name]
            try:
                return compile_model_cuda(parsed[name], os.path.join(cls.work_dir, name + ".hxx"))
            except Exception as e:
                return e

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            cls.sessions = dict(zip(cls.cases, pool.map(build, cls.cases)))

    def test_operators_and_models(self):
        for name, (module, args) in self.cases.items():
            with self.subTest(name):
                session = self.sessions[name]
                if isinstance(session, Exception):
                    raise session
                expected = module(*args).detach().numpy()
                result = session.infer(*[a.numpy() for a in args])
                self.assertEqual(result.size, expected.size)
                np.testing.assert_allclose(result, expected.flatten(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
