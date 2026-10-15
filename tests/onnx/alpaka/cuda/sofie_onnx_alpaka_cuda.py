import unittest

from alpaka_cases import cuda_cases


class sofie_onnx_alpaka_cuda(cuda_cases, unittest.TestCase):
    parser = "onnx"


if __name__ == "__main__":
    unittest.main()
