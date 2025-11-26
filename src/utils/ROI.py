import numpy as np
import cv2

def region_of_interest(img):
    mask = np.zeros_like(img)
    height, width = img.shape[:2]
    bottom_left = (0, height)
    bottom_right = (700, height)
    top_left = (0, 220)
    top_right = (640, 220)
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)