import unittest

from alpaka_cases import emit_cases


class sofie_torch_alpaka(emit_cases, unittest.TestCase):
    parser = "torch"


if __name__ == "__main__":
    unittest.main()
