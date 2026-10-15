import concurrent.futures
import contextlib
import io
import os
import unittest

import numpy as np
import torch
from generated_code_runner import compile_model, fresh_generated_dir
from operator_cases import cases
from supported import onnx_cpu

import sofie


def export_onnx(module, args, path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        torch.onnx.export(module, args, path, dynamo=True, external_data=False)


class sofie_onnx_parser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work_dir = fresh_generated_dir("onnx_parser")
        torch.manual_seed(0)
        cls.cases = {}
        for name, module, args in cases(onnx_cpu):
            cls.cases[name] = (module.eval(), args)

        parsed = {}
        for name, (module, args) in cls.cases.items():
            try:
                path = os.path.join(cls.work_dir, name + ".onnx")
                export_onnx(module, args, path)
                parsed[name] = sofie.onnx.parse(path)
            except Exception as e:
                parsed[name] = e

        def build(name):
            if isinstance(parsed[name], Exception):
                return parsed[name]
            try:
                return compile_model(parsed[name], os.path.join(cls.work_dir, name + ".hxx"))
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

    def test_check_model(self):
        import onnx
        from onnx import TensorProto, helper

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        graph = helper.make_graph([helper.make_node("NotAnOperator", ["x"], ["y"])], "g", [x], [y])
        unsupported = os.path.join(self.work_dir, "unsupported.onnx")
        onnx.save(helper.make_model(graph), unsupported)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertFalse(sofie.onnx.check_model(unsupported))
        self.assertIn("NotAnOperator", output.getvalue())

        supported = os.path.join(self.work_dir, "supported.onnx")
        export_onnx(torch.nn.ReLU().eval(), (torch.randn(2, 4),), supported)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(sofie.onnx.check_model(supported))

    def test_missing_file(self):
        with self.assertRaisesRegex(RuntimeError, "not found"):
            sofie.onnx.parse(os.path.join(self.work_dir, "missing.onnx"))


if __name__ == "__main__":
    unittest.main()
