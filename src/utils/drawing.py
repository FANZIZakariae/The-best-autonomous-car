import cv2
import numpy as np

def draw_lane_lines(img, left_line, right_line):
    line_img = np.zeros_like(img)
    if left_line is not None:
        cv2.line(line_img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
    if right_line is not None:
        cv2.line(line_img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)
    return cv2.addWeighted(img, 1.0, line_img, 0.8, 0.0)
    
    

def draw_steering_info(img, offset, steering_angle):
    height, width = img.shape[:2]
    cv2.putText(img, f"Offset: {offset:.1f}px", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, f"Steering Angle: {steering_angle:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    car_center_x = width // 2
    lane_center_x = car_center_x - int(offset)
    cv2.line(img, (car_center_x, height), (car_center_x, int(height*0.7)), (255,0,0), 2)
    cv2.line(img, (lane_center_x, height), (lane_center_x, int(height*0.7)), (0,0,255), 2)
    cv2.arrowedLine(img, (car_center_x, int(height*0.85)), 
                    (lane_center_x, int(height*0.85)), (0,255,255), 3, tipLength=0.2)
    return img
