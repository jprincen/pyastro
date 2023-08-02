import cv2
import numpy as np
import skimage as sk


def stretch_rgb_image(
    img: np.ndarray, low: float = 2.0, high: float = 98.0
) -> np.ndarray:
    """Stretch the contrast of an RGB image"""
    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    p_low, p_high = np.percentile(img_hsv[:, :, 2], (low, high))
    v_rescale = sk.exposure.rescale_intensity(
        img_hsv[:, :, 2], in_range=(p_low, p_high)
    )
    img_hsv[:, :, 2] = v_rescale
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
