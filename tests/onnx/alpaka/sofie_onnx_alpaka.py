import unittest

from alpaka_cases import emit_cases


class sofie_onnx_alpaka(emit_cases, unittest.TestCase):
    parser = "onnx"


if __name__ == "__main__":
    unittest.main()
