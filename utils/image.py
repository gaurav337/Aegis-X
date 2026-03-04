"""Image I/O utilities for Aegis-X.

Provides safe, type-hinted functions for loading and validating image files,
ensuring they are properly converted to RGB format for downstream inference.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union

# Known valid image extensions
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def is_image(path: Union[str, Path]) -> bool:
    """Checks if the given file path has a valid image extension.
    
    Args:
        path (Union[str, Path]): Path to the file.
        
    Returns:
        bool: True if the file extension is a supported image format.
    """
    # Fix: Cast to Path to prevent AttributeError if a string is passed
    return Path(path).suffix.lower() in VALID_IMAGE_EXTENSIONS

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Loads an image from disk and converts it to RGB format.
    
    Args:
        path (Union[str, Path]): Path to the image file.
        
    Returns:
        np.ndarray: The loaded image as an RGB numpy array (H, W, 3).
        
    Raises:
        FileNotFoundError: If the file does not exist or fails to load.
    """
    # Fix: Cast to Path for consistent handling
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {path}")
        
    # Fix: Use np.fromfile and cv2.imdecode to avoid cv2.imread silently failing on Unicode/non-ASCII paths
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise FileNotFoundError(f"Error reading file bytes at path {path}: {e}")
    
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load or parse image at path: {path}")
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb