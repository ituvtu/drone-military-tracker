import cv2
import numpy as np
from typing import Optional, Tuple, Any, List

class PerceptualHasher:
    """Provides utilities for computing and comparing difference hashes (dHash) of images."""
    
    @staticmethod
    def calculate_dhash(image: np.ndarray) -> np.uint64:
        """
        Computes the 64-bit difference hash (dHash) of a BGR image.
        
        Args:
            image: A BGR image represented as a numpy ndarray.
            
        Returns:
            np.uint64: The computed 64-bit perceptual hash value.
        """
        # Resize image to 9x8 (width=9, height=8)
        resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if it is in color (BGR)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        elif len(resized.shape) == 3 and resized.shape[2] == 1:
            gray = resized[:, :, 0]
        else:
            gray = resized
            
        # Compute difference between adjacent pixels in each row
        # gray has shape (8, 9)
        left = gray[:, :8]
        right = gray[:, 1:]
        
        diff = left > right # boolean matrix of shape (8, 8)
        flat_diff = diff.flatten()
        
        # Pack 64 boolean values into a single uint64
        hash_val = np.uint64(0)
        for bit in flat_diff:
            hash_val = (hash_val << np.uint64(1)) | np.uint64(bit)
            
        return hash_val

    @staticmethod
    def hamming_distance(hash1: np.uint64, hash2: np.uint64) -> int:
        """
        Calculates the Hamming distance (number of differing bits) between two 64-bit hashes.
        
        Args:
            hash1: First 64-bit hash.
            hash2: Second 64-bit hash.
            
        Returns:
            int: The Hamming distance in the range [0, 64].
        """
        xor_val = hash1 ^ hash2
        return bin(int(xor_val)).count("1")


class FrameRegistry:
    """A global registry that tracks unique frame hashes and detects duplicates."""
    
    def __init__(self, threshold: int = 8):
        """
        Initializes the registry.
        
        Args:
            threshold: Maximum Hamming distance to classify two frames as duplicates (default is 8).
        """
        self.threshold = threshold
        # List of tuples: (hash_value, frame_id)
        self.entries: List[Tuple[np.uint64, Any]] = []

    def register_frame(self, hash_value: np.uint64, frame_id: Any) -> Optional[Tuple[np.uint64, Any]]:
        """
        Checks if a similar hash exists in the registry.
        If a duplicate is found (distance <= threshold), returns (registered_hash, registered_frame_id)
        of the FIRST match found, without registering the new frame.
        If no duplicate is found, registers the new frame and returns None.
        
        Args:
            hash_value: The dHash of the new frame.
            frame_id: A unique identifier for the frame (e.g. image path or Frame object).
            
        Returns:
            Optional[Tuple[np.uint64, Any]]: The existing entry if it's a duplicate, else None.
        """
        for reg_hash, reg_id in self.entries:
            dist = PerceptualHasher.hamming_distance(hash_value, reg_hash)
            if dist <= self.threshold:
                return reg_hash, reg_id
                
        self.entries.append((hash_value, frame_id))
        return None

    def clear(self) -> None:
        """Clears all entries in the registry."""
        self.entries.clear()
