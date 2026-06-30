"""Tests for the gpustain Python module.

The test suite runs only when the `gpustain` C++ extension is importable
in the current interpreter. This is the case when the extension is
discoverable on `PYTHONPATH` (e.g. after running `PYTHONPATH=build/python`).
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np


class _GpustainTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if "gpustain" not in sys.modules and "gpustain" not in dir(cls):
            try:
                import gpustain  # noqa: F401
            except ImportError as ex:
                raise unittest.SkipTest(
                    f"gpustain C++ extension not available: {ex}") from ex

    def _synthetic_he(self, h: int = 64, w: int = 64) -> np.ndarray:
        """Generate a small synthetic H&E image as a uint8 array."""
        img = np.full((h, w, 3), [230, 200, 215], dtype=np.uint8)
        cy, cx = np.indices((h, w))
        r2     = (cx - w * 0.5) ** 2 + (cy - h * 0.5) ** 2
        core   = np.exp(-r2 / (w * h * 0.005)).astype(np.float32)
        for c, v in enumerate([80, 60, 130]):
            img[..., c] = (img[..., c] * (1 - core) + v * core).astype(np.uint8)
        return img

    def test_round_trip(self) -> None:
        import gpustain
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "in.png")
            gpustain.write_image(self._synthetic_he(), path)
            img = gpustain.read_image(path)
            self.assertEqual(img.shape[2], 3)
            self.assertEqual(img.dtype, np.uint8)


class TestVersion(unittest.TestCase):
    def test_version_is_string(self) -> None:
        try:
            import gpustain
        except ImportError:
            self.skipTest("gpustain not available")
        self.assertIsInstance(gpustain.__version__, str)


if __name__ == "__main__":
    unittest.main()
