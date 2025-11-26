import numpy as np
import cv2

def get_auto_canny_thresholds(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]
    low_thresh = np.searchsorted(cdf_normalized, 0.10)
    high_thresh = np.searchsorted(cdf_normalized, 0.90)
    return int(low_thresh), int(high_thresh)
