import unittest

from alpaka_cases import emit_cases


class sofie_keras_alpaka(emit_cases, unittest.TestCase):
    parser = "keras"


if __name__ == "__main__":
    unittest.main()
