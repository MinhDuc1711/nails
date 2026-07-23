import unittest

import numpy as np

from web_app import app, analyze_nails, normalize_rgb


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_renders(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload a nail image', response.data)
        self.assertIn(b'Color analysis', response.data)

    def test_analysis_handles_component_mask_shape(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        image[8:24, 8:24] = [255, 0, 0]
        pred_mask = np.zeros((40, 40, 2), dtype=np.float32)
        pred_mask[8:24, 8:24, 0] = 1.0

        analysis = analyze_nails(image, pred_mask, 0)

        self.assertIn('rgb_graph', analysis)
        self.assertIn('nails', analysis)
        self.assertGreaterEqual(len(analysis['nails']), 1)

    def test_normalize_rgb_keeps_values_between_zero_and_one(self):
        normalized = normalize_rgb([255, 128, 0])
        self.assertAlmostEqual(normalized[0], 1.0)
        self.assertAlmostEqual(normalized[1], 0.50196, places=4)
        self.assertAlmostEqual(normalized[2], 0.0)

    def test_analysis_returns_at_most_five_fingers(self):
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        pred_mask = np.zeros((60, 60, 2), dtype=np.float32)
        for i in range(8):
            x = 4 + i * 6
            image[8:20, x:x + 8] = [200, 150, 100]
            pred_mask[8:20, x:x + 8, 0] = 1.0

        analysis = analyze_nails(image, pred_mask, 0)
        self.assertLessEqual(len(analysis['nails']), 5)


if __name__ == '__main__':
    unittest.main()
