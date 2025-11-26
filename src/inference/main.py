import cv2
import numpy as np
from collections import deque
from utils.canny import get_auto_canny_thresholds 
from utils.ROI import region_of_interest
from utils.slope import average_slope_intercept
from utils.drawing import *
from utils.pwm_steering import PWMServo
from utils.esc import ESC


# --------------------
# Parameters
# --------------------
VIDEO_PATH = "/home/user/Desktop/MyCar/test/real_car.mp4"
SHOW_WINDOW_SIZE = (640, 360)
LINE_HISTORY = 5
Kp = 0.05

# --------------------
# Lane Smoothing Memory
# --------------------
left_lines_history = deque(maxlen=LINE_HISTORY)
right_lines_history = deque(maxlen=LINE_HISTORY)

# --------------------
# Steering Data Storage
# --------------------
steering_log = []

# --------------------
# Main Pipeline
# --------------------
def main():
    # === NEW: Initialize servo ===
    servo = PWMServo(pin=17)
    esc = ESC(pin=2)
        
    # Start forward movement at very slow speed
    esc.set_speed(1560)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    frame_num = 0
    smoothed_angle = 90
    alpha = 0.15    # smaller = smoother (0.1–0.2 recommended)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            frame = cv2.resize(frame, SHOW_WINDOW_SIZE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            low_t, high_t = get_auto_canny_thresholds(gray)
            edges = cv2.Canny(gray, low_t, high_t)

            roi_edges = region_of_interest(edges)

            lines = cv2.HoughLinesP(
                roi_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=40,
                maxLineGap=150
            )

            left_line, right_line = average_slope_intercept(lines)

            if left_line is not None:
                left_lines_history.append(left_line)
            if right_line is not None:
                right_lines_history.append(right_line)

            left_avg_line = np.mean(left_lines_history, axis=0).astype(int) if left_lines_history else None
            right_avg_line = np.mean(right_lines_history, axis=0).astype(int) if right_lines_history else None

            height, width = frame.shape[:2]

            if left_avg_line is not None and right_avg_line is not None:
                left_x_bottom = left_avg_line[0]
                right_x_bottom = right_avg_line[0]
                lane_center = (left_x_bottom + right_x_bottom) / 2
                car_center = width / 2
                offset = car_center - lane_center
                steering_angle = Kp * offset
            else:
                offset = 0
                steering_angle = 0

            # Print steering info
            print(f"Frame {frame_num}: Offset={offset:.1f}, Steering Angle={steering_angle:.1f}")

            # === NEW DISCRETE STEERING LEVEL SYSTEM ===
            if -1 <= steering_angle <= 1:
                servo_angle = 90               # center
            elif 1 < steering_angle <= 3:
                servo_angle = 80             # small right
            elif steering_angle > 3:
                servo_angle = 60             # big right
            elif -3 <= steering_angle < -1:
                servo_angle = 100              # small left
            elif steering_angle < -3:
                servo_angle = 120              # big left

            # Optional: small smoothing so servo does not snap
            smoothed_angle = smoothed_angle * (1 - alpha) + servo_angle * alpha

            servo.set_angle(smoothed_angle)
            
            # >>> NEW: KEEP MOTOR MOVING FORWARD
            esc.update()


            lane_frame = draw_lane_lines(frame, left_avg_line, right_avg_line)
            lane_frame = draw_steering_info(lane_frame, offset, steering_angle)

            combined = np.hstack((cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR), lane_frame))
            cv2.imshow("ROI Edges + Lane Detection + Steering", combined)

            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                break

    finally:
        # === NEW: Clean up servo ===
        servo.cleanup()
        esc.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()

