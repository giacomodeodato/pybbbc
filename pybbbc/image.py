"""
Functions for working with images
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def correct_illumination(
    images: np.ndarray, sigma=500, min_percentile=0.02
) -> np.ndarray:
    img_avg = images.mean(axis=0)
    img_mask = gaussian_filter(img_avg.astype(np.float32), sigma=sigma).astype(
        np.float16
    )
    robust_min = np.percentile(img_mask[img_mask > 0], min_percentile)
    img_mask[img_mask < robust_min] = robust_min
    img_mask = img_mask / robust_min
    return images / img_mask


def scale_pixel_intensity(images: np.ndarray) -> np.ndarray:
    """
    Scale image intensities from [0...1] such that 0 represents the 0.1th
    percentile and 1 the 99.9th percentile intensity in the original image.

    Intensities are clipped from [0, inf].

    Args:
        images:

    Returns:
        Copy of `images` with pixel intensities rescaled.
    """
    low = np.percentile(images, 0.1)
    high = np.percentile(images, 99.9)

    images = (images - low) / (high - low)

    return np.clip(images, 0, None)
