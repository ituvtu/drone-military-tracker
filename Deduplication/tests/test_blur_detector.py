import unittest
import numpy as np
import cv2
from Deduplication.blur_detector import BlurDetector

class TestBlurDetector(unittest.TestCase):
    
    def test_uniform_image(self):
        # A completely uniform image should have zero Laplacian variance
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        variance = BlurDetector.get_blur_variance(img)
        self.assertEqual(variance, 0.0)

    def test_sharp_vs_blurry(self):
        # Create a sharp image with high contrast edges
        img_sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a high-contrast white rectangle
        cv2.rectangle(img_sharp, (25, 25), (75, 75), (255, 255, 255), -1)
        
        # Create a blurry version using Gaussian Blur
        img_blurry = cv2.GaussianBlur(img_sharp, (9, 9), 0)
        
        var_sharp = BlurDetector.get_blur_variance(img_sharp)
        var_blurry = BlurDetector.get_blur_variance(img_blurry)
        
        # Sharp image must have a significantly higher Laplacian variance than blurry image
        self.assertGreater(var_sharp, var_blurry)
        self.assertGreater(var_blurry, 0.0)

    def test_grayscale_input(self):
        # Sharp image in grayscale
        img_sharp_color = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img_sharp_color, (25, 25), (75, 75), (255, 255, 255), -1)
        img_sharp_gray = cv2.cvtColor(img_sharp_color, cv2.COLOR_BGR2GRAY)
        
        var_color = BlurDetector.get_blur_variance(img_sharp_color)
        var_gray = BlurDetector.get_blur_variance(img_sharp_gray)
        
        # Color and grayscale variants of the same image should have identical or extremely close variance
        self.assertAlmostEqual(var_color, var_gray, places=4)

if __name__ == "__main__":
    unittest.main()
