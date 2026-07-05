import unittest
import numpy as np

from app import preprocess_face_image


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_face_image_returns_expected_shape_and_range(self):
        face = np.zeros((120, 120), dtype=np.uint8)
        cv2 = __import__('cv2')
        cv2.rectangle(face, (20, 20), (100, 100), 255, -1)

        image = preprocess_face_image(face)

        self.assertEqual(image.shape, (1, 48, 48, 1))
        self.assertTrue(np.all(image >= 0.0))
        self.assertTrue(np.all(image <= 1.0))


if __name__ == '__main__':
    unittest.main()
