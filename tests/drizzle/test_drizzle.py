import numpy as np

from drizzle import Drizzle
import unittest

class TestDrizzle(unittest.TestCase):
    def test_identity_transform(self):
        drizzle = Drizzle(10, 10, 1)
        image = np.ones((5, 5, 1))
        tx = np.identity(3)
        drizzle.add_image(tx, image)
        expected = np.zeros((2, 2, 1))
        expected[0, 0, 0] = 1
        np.testing.assert_array_equal(drizzle.image[0:2, 0:2], expected)
    def test_out_of_bounds(self):
        # Put input out of bounds, so nothing should be added
        drizzle = Drizzle(5, 10, 1)
        image = np.ones((2, 2, 1))
        tx = np.identity(3)
        tx[0, 2] = 10
        drizzle.add_image(tx, image)
        np.testing.assert_array_equal(drizzle.image, np.zeros((5, 10, 1)))
        
    def test_rgb(self):
        drizzle = Drizzle(10, 10, 3)
        image = np.ones((5, 5, 3))
        tx = np.identity(3)
        drizzle.add_image(tx, image)
        expected = np.zeros((2, 2, 3))
        expected[0, 0, :] = 1
        np.testing.assert_array_equal(drizzle.image[0:2, 0:2], expected)

if __name__ == '__main__':
    unittest.main()
