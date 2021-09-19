"""
Functions for working with images
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def correct_illumination(
    images: np.ndarray, sigma=500, min_percentile=0.02
) -> np.ndarray:
    # calculate average of images belonging to the same site and plate
    img_avg = images.mean(axis=0)

    # apply Gaussian filter
    img_mask = gaussian_filter(img_avg.astype(np.float32), sigma=sigma).astype(
        np.float16
    )

    # calculate robust minimum
    robust_min = np.percentile(img_mask[img_mask > 0], min_percentile)

    # clip and scale pixel intensities
    img_mask[img_mask < robust_min] = robust_min
    img_mask = img_mask / robust_min

    # return corrected images
    return images / img_mask


def scale_pixel_intensity(images: np.ndarray) -> np.ndarray:
    """
    Scale image intensities from [0...1] such that 0 represents the 0.1th
    percentile and 1 the 99.9th percentile intensity in the original image.

    Intensities are clipped from [0, inf].

    Returns:
        Copy of `images` with pixel intensities rescaled.
    """
    low = np.percentile(images, 0.1)
    high = np.percentile(images, 99.9)

    images = (images - low) / (high - low)

    return np.clip(images, 0, None)
