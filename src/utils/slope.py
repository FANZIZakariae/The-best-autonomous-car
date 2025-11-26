import cv2
import numpy as np

SHOW_WINDOW_SIZE = (640, 360)

def average_slope_intercept(lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if slope < -0.3:
                left_lines.append((slope, intercept))
            elif slope > 0.3:
                right_lines.append((slope, intercept))

    def make_line(slope_intercept_list):
        if len(slope_intercept_list) == 0:
            return None
        slope, intercept = np.mean(slope_intercept_list, axis=0)
        y1 = SHOW_WINDOW_SIZE[1]
        y2 = int(y1 * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]

    left_avg = make_line(left_lines)
    right_avg = make_line(right_lines)
    return left_avg, right_avg
