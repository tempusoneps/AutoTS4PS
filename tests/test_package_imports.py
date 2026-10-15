import unittest
import sys
import os

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

class TestPackageImports(unittest.TestCase):
    def test_autots_import(self):
        import autots
        self.assertIsNotNone(autots.__file__)

    def test_autots_core_classes(self):
        from autots import AutoTS, GeneralTransformer
        self.assertTrue(callable(AutoTS))
        self.assertTrue(callable(GeneralTransformer))

if __name__ == '__main__':
    unittest.main()
