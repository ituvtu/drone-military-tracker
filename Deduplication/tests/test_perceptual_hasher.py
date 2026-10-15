import unittest
import numpy as np
from Deduplication.perceptual_hasher import PerceptualHasher, FrameRegistry

class TestPerceptualHasher(unittest.TestCase):
    
    def test_hamming_distance(self):
        # Test basic distances
        h1 = np.uint64(0b0)
        h2 = np.uint64(0b0)
        self.assertEqual(PerceptualHasher.hamming_distance(h1, h2), 0)
        
        h1 = np.uint64(0b1)
        h2 = np.uint64(0b0)
        self.assertEqual(PerceptualHasher.hamming_distance(h1, h2), 1)
        
        h1 = np.uint64(0b1100)
        h2 = np.uint64(0b1010)
        self.assertEqual(PerceptualHasher.hamming_distance(h1, h2), 2)
        
        # Test full 64-bit distances
        h1 = np.uint64(0xFFFFFFFFFFFFFFFF)
        h2 = np.uint64(0x0000000000000000)
        self.assertEqual(PerceptualHasher.hamming_distance(h1, h2), 64)

    def test_calculate_dhash_equivalence(self):
        # Create a simple random BGR image
        np.random.seed(42)
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Identical image should have the same hash
        hash1 = PerceptualHasher.calculate_dhash(img1)
        hash2 = PerceptualHasher.calculate_dhash(img1.copy())
        
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, np.uint64)

    def test_calculate_dhash_sensitivity(self):
        # Create a base image
        img_base = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2_mock_draw(img_base, (10, 10), (40, 40), (255, 255, 255))
        
        # Slightly modified image (slight shift or noise)
        img_noisy = img_base.copy()
        img_noisy[img_noisy > 0] = 250 # change intensity slightly
        
        # Completely different image (circle at another spot)
        img_diff = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2_mock_draw(img_diff, (70, 70), (90, 90), (255, 255, 255))
        
        h_base = PerceptualHasher.calculate_dhash(img_base)
        h_noisy = PerceptualHasher.calculate_dhash(img_noisy)
        h_diff = PerceptualHasher.calculate_dhash(img_diff)
        
        dist_noisy = PerceptualHasher.hamming_distance(h_base, h_noisy)
        dist_diff = PerceptualHasher.hamming_distance(h_base, h_diff)
        
        # Noisy image should be very close (distance close to 0)
        self.assertLessEqual(dist_noisy, 4)
        # Different image should be far (distance much larger than 8)
        self.assertGreater(dist_diff, 8)

    def test_frame_registry(self):
        registry = FrameRegistry(threshold=8)
        
        # Create some mock hashes
        hash_base = np.uint64(0x0F0F0F0F0F0F0F0F)
        # Distance = 4 (differ in 4 bits)
        hash_near = np.uint64(0x0F0F0F0F0F0F0F0E) # change last hex digit F (1111) to E (1110) - 1 bit
        # Let's change 3 more bits in first hex digit: 0 (0000) to 7 (0111) - 3 bits. Total 4 bits difference.
        hash_near = np.uint64(0x7F0F0F0F0F0F0F0E)
        
        # Distance = 16 (differ in 16 bits)
        hash_far = np.uint64(0xF0F0F0F0F0F0F0F0)
        
        # Base frame
        res1 = registry.register_frame(hash_base, "frame_base")
        self.assertIsNone(res1)
        
        # Near frame (duplicate) -> should return base entry
        res2 = registry.register_frame(hash_near, "frame_near")
        self.assertIsNotNone(res2)
        self.assertEqual(res2[0], hash_base)
        self.assertEqual(res2[1], "frame_base")
        
        # Far frame (unique) -> should register and return None
        res3 = registry.register_frame(hash_far, "frame_far")
        self.assertIsNone(res3)
        
        # Verify that clear removes entries
        registry.clear()
        self.assertEqual(len(registry.entries), 0)
        res4 = registry.register_frame(hash_near, "frame_near")
        self.assertIsNone(res4)

def cv2_mock_draw(img, pt1, pt2, color):
    # Standard NumPy drawing equivalent to cv2.rectangle
    # pt is (x, y) which maps to col, row
    r1, c1 = pt1[1], pt1[0]
    r2, c2 = pt2[1], pt2[0]
    img[r1:r2+1, c1:c2+1] = color

if __name__ == "__main__":
    unittest.main()
