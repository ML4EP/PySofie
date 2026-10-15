import os
import shutil
import unittest

import numpy as np

import sofie
from generated_code_runner import compile_model, generated_dir

# Test SOFIE PyTorch parser (sofie.pytorch.parse) by
# parsing models exported with torch.export, generating and compiling the
# inference code and comparing SOFIE results with the outputs of the
# original PyTorch models.
work_dir = generated_dir("pytorch_parser_models")


def generate_pytorch_models(dst_dir):
    import torch
    import torch.nn as nn

    def train_and_save(model, x, y, name, n_iterations):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(n_iterations):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        torch.export.save(torch.export.export(model, (x,)), f"{dst_dir}/{name}.pt2")

    # Sequential model
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 6),
        nn.SELU(),
    )
    train_and_save(model, torch.randn(2, 4), torch.randn(2, 6), "PyTorchModelSequential", 2000)

    # Module model
    class module_model(nn.Module):
        def __init__(self):
            super(module_model, self).__init__()
            self.fc1 = nn.Linear(6, 36)
            self.fc2 = nn.Linear(36, 12)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            x = torch.transpose(x, 1, 0)
            return x

    train_and_save(module_model(), torch.randn(2, 6), torch.randn(12, 2), "PyTorchModelModule", 2000)

    # Convolution model
    model = nn.Sequential(
        nn.Conv2d(6, 5, 3, stride=2),
        nn.ReLU(),
    )
    train_and_save(model, torch.randn(5, 6, 5, 5), torch.randn(5, 5, 2, 2), "PyTorchModelConvolution", 100)


class sofie_py_torch_parser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        print("Generating PyTorch models for testing")
        generate_pytorch_models(work_dir)

    def generate_and_test_inference(self, model_name, input_tensor, atol):
        import torch

        model_path = f"{work_dir}/{model_name}.pt2"

        # Parse the exported program. The input tensor shapes (and thus the
        # batch size) are the ones the model was exported with.
        rmodel = sofie.pytorch.parse(model_path)

        # Generate and compile SOFIE inference code
        header_path = f"{work_dir}/{model_name}.hxx"
        print(f"Compiling SOFIE model {model_name}")
        inference_session = compile_model(rmodel, header_path, batch_size=input_tensor.shape[0])

        sofie_inference_result = inference_session.infer(input_tensor)

        # Evaluate the original model with PyTorch on the same input
        torch_model = torch.export.load(model_path).module()
        torch_inference_result = torch_model(torch.from_numpy(input_tensor)).detach().numpy()

        # Compare the output tensor sizes and values
        self.assertEqual(len(sofie_inference_result), torch_inference_result.size)
        np.testing.assert_allclose(
            sofie_inference_result,
            torch_inference_result.flatten(),
            atol=atol,
            rtol=0.0,  # explicitly disable relative tolerance (numpy uses |a - b| <= atol + rtol * |b|)
        )

    def test_sequential_model(self):
        input_tensor = np.array(
            [[-1.6207, 0.6133, 0.5058, -1.2560], [-0.7750, -1.6701, 0.8171, -0.2858]], dtype=np.float32
        )
        self.generate_and_test_inference("PyTorchModelSequential", input_tensor, atol=1e-6)

    def test_module_model(self):
        input_tensor = np.array(
            [
                [0.5516, 0.3585, -0.4854, -1.3884, 0.8057, -0.9449],
                [0.5626, -0.6466, -1.8818, 0.4736, 1.1102, 1.8694],
            ],
            dtype=np.float32,
        )
        self.generate_and_test_inference("PyTorchModelModule", input_tensor, atol=1e-6)

    def test_decomposed_module_model(self):
        # decompose=True lowers linear to t + addmm, as AOTInductor does
        import torch

        input_tensor = np.random.default_rng(0).standard_normal((2, 6)).astype(np.float32)
        model_path = f"{work_dir}/PyTorchModelModule.pt2"
        rmodel = sofie.pytorch.parse(model_path, decompose=True)
        session = compile_model(rmodel, f"{work_dir}/PyTorchModelModuleDecomposed.hxx", batch_size=2)
        expected = torch.export.load(model_path).module()(torch.from_numpy(input_tensor)).detach().numpy()
        np.testing.assert_allclose(session.infer(input_tensor), expected.flatten(), atol=1e-6, rtol=0.0)

    def test_aot_compiled_package(self):
        # A package with the AOT-compiled model next to its exported program can be parsed,
        # a plain AOTInductor package cannot.
        import torch
        from torch.export.pt2_archive._package import package_pt2

        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh()).eval()
        example = (torch.randn(3, 4),)
        ep = torch.export.export(model, example)
        files = torch._inductor.aot_compile(ep.module(), example, options={"aot_inductor.package": True})
        combined = f"{work_dir}/combined.pt2"
        package_pt2(combined, exported_programs=ep, aoti_files=files)

        session = compile_model(sofie.pytorch.parse(combined), f"{work_dir}/combined.hxx", batch_size=3)
        expected = torch._inductor.aoti_load_package(combined)(*example).detach().numpy()
        np.testing.assert_allclose(session.infer(example[0].numpy()), expected.flatten(), atol=1e-6, rtol=0.0)

        plain = torch._inductor.aoti_compile_and_package(ep, package_path=f"{work_dir}/plain_aoti.pt2")
        with self.assertRaisesRegex(RuntimeError, "plain AOTInductor package"):
            sofie.pytorch.parse(plain)

    def test_custom_operator(self):
        # A custom operator is opaque: it only parses with a decomposition into ATen operators
        import torch

        @torch.library.custom_op("sofie_test::scale_by_three", mutates_args=())
        def scale_by_three(x: torch.Tensor) -> torch.Tensor:
            return x * 3.0

        @scale_by_three.register_fake
        def _(x):
            return torch.empty_like(x)

        class custom_op_model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 3)

            def forward(self, x):
                return torch.relu(scale_by_three(self.fc(x)))

        model = custom_op_model().eval()
        input_tensor = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)
        ep = torch.export.export(model, (torch.from_numpy(input_tensor),))

        # without a decomposition the parser reports the operator
        with self.assertRaisesRegex(RuntimeError, "sofie_test.scale_by_three.default"):
            sofie.pytorch.parse(ep)
        self.assertEqual(
            [o["op"] for o in sofie.pytorch.list_operators(ep) if not o["supported"]], ["sofie_test.scale_by_three.default"]
        )
        with self.assertRaisesRegex(ValueError, "decompose=True"):
            sofie.pytorch.parse(ep, decompose=False, decompositions={})

        # with one, the operator breaks down into ATen operators
        decompositions = {torch.ops.sofie_test.scale_by_three.default: lambda x: x * 3.0}
        self.assertTrue(all(o["supported"] for o in sofie.pytorch.list_operators(ep, decompositions=decompositions)))
        session = compile_model(sofie.pytorch.parse(ep, decompositions=decompositions), f"{work_dir}/custom.hxx", batch_size=2)
        expected = model(torch.from_numpy(input_tensor)).detach().numpy()
        np.testing.assert_allclose(session.infer(input_tensor), expected.flatten(), atol=1e-6, rtol=0.0)

    def test_convolution_model(self):
        input_tensor = np.arange(1, 751, dtype=np.float32).reshape(5, 6, 5, 5)
        self.generate_and_test_inference("PyTorchModelConvolution", input_tensor, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
