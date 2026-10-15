import unittest

from alpaka_cases import cuda_cases


class sofie_keras_alpaka_cuda(cuda_cases, unittest.TestCase):
    parser = "keras"


if __name__ == "__main__":
    unittest.main()
