
import unittest
from image_collection import ImageCollection, RawImageCollection
import numpy as np


class TestImageCollection(unittest.TestCase):
    def test_creation(self):
        ones = np.ones((2, 2))
        ic = RawImageCollection([ones, ones])
        self.assertEqual(len(ic), 2)
        for img in ic:
            self.assertEqual(np.max(img), np.min(img))
            self.assertEqual(np.max(img), 1)

    def test_add(self):
        ones = np.ones((2, 2))
        ic = RawImageCollection([ones, ones])
        ic = ic + ones
        self.assertEqual(len(ic), 2)
        for img in ic:
            self.assertEqual(np.max(img), np.min(img))
            self.assertEqual(np.max(img), 2)

    def test_sub(self):
        ones = np.ones((2, 2))
        ic = RawImageCollection([ones, ones])
        ic = ic - ones
        self.assertEqual(len(ic), 2)
        for img in ic:
            self.assertEqual(np.max(img), np.min(img))
            self.assertEqual(np.max(img), 0)

    def test_mul(self):
        ones = np.ones((2, 2))
        ic = RawImageCollection([ones, ones])
        # Arrays by scalar
        ic = ic * 3
        self.assertEqual(len(ic), 2)
        for img in ic:
            self.assertEqual(np.max(img), np.min(img))
            self.assertEqual(np.max(img), 3)
        # Multiply arrays by array
        ic = ic * ic[0]
        self.assertEqual(len(ic), 2)
        for img in ic:
            self.assertEqual(np.max(img), np.min(img))
            self.assertEqual(np.max(img), 9)

    def test_div(self):
        ones = 2*np.ones((2, 2))
        ic = RawImageCollection([ones, ones])
        # Arrays by scalar
        ic = ic / 2
        self.assertEqual(len(ic), 2)
        for img in ic:
            self.assertEqual(np.max(img), np.min(img))
            self.assertEqual(np.max(img), 1.0)
        # Multiply arrays by array
        ic = ic / ic[0]
        self.assertEqual(len(ic), 2)
        for img in ic:
            self.assertEqual(np.max(img), np.min(img))
            self.assertEqual(np.max(img), 1)

if __name__ == '__main__':
    unittest.main()
