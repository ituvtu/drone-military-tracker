import cv2
import numpy as np

class BlurDetector:
    """Detects frame blur and measures image sharpness using the variance of the Laplacian."""
    
    @staticmethod
    def get_blur_variance(image: np.ndarray) -> float:
        """
        Calculates the variance of the Laplacian for the given image.
        Higher values indicate a sharper image, while lower values indicate a blurry image.
        
        Args:
            image: BGR or Grayscale image represented as a numpy array.
            
        Returns:
            float: The variance of the Laplacian.
        """
        # Convert BGR to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            gray = image
            
        # Compute the Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance
        variance = float(laplacian.var())
        
        return variance
