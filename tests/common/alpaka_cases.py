import os
import unittest

import numpy as np

import sofie
from generated_code_runner import (
    alpaka_enabled,
    check_alpaka_header,
    compile_model_cuda,
    cuda_available,
    fresh_generated_dir,
)
from parsed_mlp import parsed_mlp


class emit_cases:
    parser = None

    @unittest.skipIf(sofie.with_alpaka, "this build can generate alpaka code")
    def test_off_by_default(self):
        model = sofie.RModel("m", "now")
        with self.assertRaisesRegex(RuntimeError, "-C alpaka=cuda"):
            model.generate_gpu_alpaka()

    @unittest.skipUnless(sofie.with_alpaka, "needs a build for an alpaka backend")
    def test_generate(self):
        tmp = fresh_generated_dir(self.parser, "alpaka_generate")
        rmodel, _, _ = parsed_mlp(tmp, self.parser)
        rmodel.generate_gpu_alpaka(batch_size=2)
        header = os.path.join(tmp, "mlp.hxx")
        rmodel.output_generated(header)
        with open(header) as f:
            code = f.read()
        self.assertIn("#include <alpaka/alpaka.hpp>", code)
        self.assertIn("#include <sofieBLAS/sofieBLAS.hpp>", code)
        self.assertIn("struct Session", code)
        self.assertNotIn("sgemm_", code)

    @unittest.skipUnless(sofie.with_alpaka, "needs a build for an alpaka backend")
    def test_kernel_only(self):
        tmp = fresh_generated_dir(self.parser, "alpaka_kernel_only")
        rmodel, _, _ = parsed_mlp(tmp, self.parser)
        rmodel.generate_gpu_alpaka(sofie.Options.kKernelOnly, batch_size=2)
        header = os.path.join(tmp, "mlp.hxx")
        rmodel.output_generated(header)
        with open(header) as f:
            code = f.read()
        self.assertIn("#include <alpaka/alpaka.hpp>", code)
        self.assertNotIn("struct Session", code)

    @unittest.skipUnless(alpaka_enabled(), "needs a build for an alpaka backend, alpaka and sofieBLAS")
    def test_compiles_against_alpaka(self):
        for kernel_only in (False, True):
            with self.subTest(kernel_only=kernel_only):
                tmp = fresh_generated_dir(self.parser, "alpaka_compiles", "kernel_only" if kernel_only else "full")
                rmodel, _, _ = parsed_mlp(tmp, self.parser)
                if kernel_only:
                    rmodel.generate_gpu_alpaka(sofie.Options.kKernelOnly, batch_size=2)
                else:
                    rmodel.generate_gpu_alpaka(batch_size=2)
                header = os.path.join(tmp, "mlp.hxx")
                rmodel.output_generated(header)
                check_alpaka_header(header)


@unittest.skipUnless(cuda_available(), "needs a build for the cuda backend, a CUDA GPU, nvcc, alpaka and sofieBLAS")
class cuda_cases:
    parser = None

    def test_mlp(self):
        tmp = fresh_generated_dir(self.parser, "cuda_mlp")
        rmodel, x, expected = parsed_mlp(tmp, self.parser)
        session = compile_model_cuda(rmodel, os.path.join(tmp, "mlp.hxx"), batch_size=2)
        np.testing.assert_allclose(session.infer(x), expected, atol=1e-4, rtol=1e-4)
