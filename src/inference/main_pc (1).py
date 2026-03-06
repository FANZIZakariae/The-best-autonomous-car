#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import threading
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import time

# ---------------------------------------------------------
# 1. SETUP PATHS
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ---------------------------------------------------------
# 2. IMPORTS
# ---------------------------------------------------------
try:
    from utils.canny import get_auto_canny_thresholds 
    # We override the other utils manually below for better control
    from utils.drawing import draw_lane_lines
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 3. PARAMETERS
# ---------------------------------------------------------
VIDEO_PATH = "test/Road_test.mp4" 
SHOW_WINDOW_SIZE = (640, 360) 

# Lane Settings
LINE_HISTORY = 5
Kp = 0.05
ALPHA_STEER = 0.15 

# AI Models
YOLO_OBJECTS_MODEL = "models/yolov8n.pt"
YOLO_SIGNS_MODEL = "models/best.pt"
CNN_SIGNS_MODEL = "models/GTSRB_43classes.h5"

# ---------------------------------------------------------
# 4. CUSTOM FUNCTIONS (ROI & Slope)
# ---------------------------------------------------------
def strict_region_of_interest(img):
    """
    Applies the mask for CALCULATION only.
    """
    height, width = img.shape[:2]
    
    # Polygon Logic (Widened for curves)
    polygons = np.array([
        [
            (int(width * 0.1), height),
            (int(width * 0.7), height),
            (int(width * 0.5), int(height * 0.58)),
            (int(width * 0.45), int(height * 0.58))
        ]
    ])
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def average_slope_intercept(lines):
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2: continue
            
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            if slope < -0.3: 
                left_fit.append((slope, intercept))
            elif slope > 0.3:
                right_fit.append((slope, intercept))

    left_avg_line = make_coordinates(lines, np.average(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_avg_line = make_coordinates(lines, np.average(right_fit, axis=0)) if len(right_fit) > 0 else None
    
    return left_avg_line, right_avg_line

def make_coordinates(image, line_parameters):
    if line_parameters is None: return None
    slope, intercept = line_parameters
    try:
        y1 = 360 
        y2 = int(y1 * 0.6) 
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    except:
        return None

# ---------------------------------------------------------
# 5. FANCY HUD FUNCTIONS
# ---------------------------------------------------------
def draw_transparent_rect(img, pts, color, alpha=0.5):
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts)], color)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_hud(frame, steering_angle, throttle, message, pwm_val):
    h, w = frame.shape[:2]
    
    # 1. Top Status Bar
    frame = draw_transparent_rect(frame, [(0, 0), (w, 0), (w, 50), (0, 50)], (20, 20, 40), 0.7)
    
    # 2. Throttle Bar (Right)
    bar_h, bar_w, bar_x, bar_y = 200, 20, w - 40, h // 2 - 100
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), 2)
    fill_ratio = min(1.0, throttle / 0.3) 
    fill_h = int(bar_h * fill_ratio)
    col = (0, 255, 0) if fill_ratio < 0.8 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x + 2, bar_y + bar_h - fill_h), (bar_x + bar_w - 2, bar_y + bar_h), col, -1)
    
    # 3. Steering Arc (Bottom)
    center_x, center_y, radius = w // 2 - 40, h - 40, 100
    #cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 180, 360, (50, 50, 50), 2)
    #needle_angle = 270 + (steering_angle * 20)
    #end_x = int(center_x + (radius - 10) * np.cos(np.radians(needle_angle)))
    #end_y = int(center_y + (radius - 10) * np.sin(np.radians(needle_angle)))
    #cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 3)

    # 4. Text
    def glow(img, txt, pos, s, c):
        cv2.putText(img, txt, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), 3)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, s, c, 1)

    status_col = (0, 255, 0) if "libre" in message else (0, 0, 255)
    glow(frame, f"AI: {message}", (20, 35), 0.7, status_col)
    
    steer_txt = "CENTER"
    if steering_angle > 1: steer_txt = "RIGHT >>"
    elif steering_angle < -1: steer_txt = "<< LEFT"
    glow(frame, f"STR: {steer_txt}", (w - 200, 35), 0.6, (255, 200, 0))
    
    return frame

# ---------------------------------------------------------
# 6. GLOBAL VARS & MODELS
# ---------------------------------------------------------
left_lines_history = deque(maxlen=LINE_HISTORY)
right_lines_history = deque(maxlen=LINE_HISTORY)

CNN_AVAILABLE = None
tf = None
_cnn_model_cache = None
_cnn_loading = False

IDX_TO_LABEL = {
    0: "Vitesse 20", 1: "Vitesse 30", 2: "Vitesse 50", 14: "Stop", 
    33: "Tourner Droite", 34: "Tourner Gauche", 26: "Feu Tricolore"
}

# ---------------------------------------------------------
# 7. AI LOGIC
# ---------------------------------------------------------
def load_cnn_lazy():
    global CNN_AVAILABLE, tf, _cnn_loading, _cnn_model_cache
    if CNN_AVAILABLE is False or _cnn_loading or _cnn_model_cache is not None: return None
    _cnn_loading = True
    def load_bg():
        global CNN_AVAILABLE, tf, _cnn_model_cache, _cnn_loading
        try:
            import tensorflow as tf_mod
            tf = tf_mod
            tf.get_logger().setLevel('ERROR')
            if os.path.exists(CNN_SIGNS_MODEL):
                _cnn_model_cache = tf.keras.models.load_model(CNN_SIGNS_MODEL)
                CNN_AVAILABLE = True
                print("\n✅ CNN Loaded")
            else: CNN_AVAILABLE = False
        except: CNN_AVAILABLE = False
        finally: _cnn_loading = False
    threading.Thread(target=load_bg, daemon=True).start()

def classify_sign(cnn_model, crop_bgr):
    if not CNN_AVAILABLE or cnn_model is None: return "Panneau", 0.5, True
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb).resize((32, 32))
    x = np.array(img, dtype=np.float32).reshape((1, 32, 32, 3)) / 255.0
    proba = cnn_model.predict(x, verbose=0)[0]
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    return (IDX_TO_LABEL.get(idx, f"Class {idx}"), conf, False) if conf >= 0.75 else ("Inconnu", conf, True)

def detect_traffic_light_color(crop_img):
    if crop_img.size == 0: return "Inconnu"
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mask_r = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    )
    mask_g = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    r, g = cv2.countNonZero(mask_r), cv2.countNonZero(mask_g)
    if r > g and r > 20: return "Rouge"
    if g > r and g > 20: return "Vert"
    return "Inconnu"

def calculate_throttle(detections):
    target, msg = 0.30, "Voie libre"
    emer, slow = False, False
    for lbl, _, _, extra in detections:
        lbl = lbl.lower()
        if 'stop' in lbl or 'person' in lbl or extra.get('color') == 'Rouge':
            emer = True; msg = "🛑 EMERGENCY"
        elif any(x in lbl for x in ['car', 'truck', 'vitesse 30']):
            slow = True; msg = "⚠️ CAUTION"
    if emer: return 0.0, msg
    if slow: return 0.15, msg
    return target, msg

# ---------------------------------------------------------
# 8. MAIN PIPELINE
# ---------------------------------------------------------
def main():
    print("🚀 STARTING SYNCHRONIZED SIMULATION")
    video_source = VIDEO_PATH if os.path.exists(VIDEO_PATH) else 0
    
    yolo_objects = YOLO(YOLO_OBJECTS_MODEL)
    yolo_signs = YOLO(YOLO_SIGNS_MODEL) if os.path.exists(YOLO_SIGNS_MODEL) else yolo_objects
    
    cap = cv2.VideoCapture(video_source)
    smoothed_angle = 90
    global _cnn_model_cache

    try:
        while True:
            ret, frame = cap.read()
            
            # --- VIDEO LOOP LOGIC ---
            if not ret: 
                if isinstance(video_source, str): 
                    print("🔄 Video Ended - Restarting Loop")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: break

            frame = cv2.resize(frame, SHOW_WINDOW_SIZE)
            
            # --- LANE DETECTION ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            low_t, high_t = get_auto_canny_thresholds(gray)
            edges = cv2.Canny(gray, low_t, high_t)
            
            # WE USE ROI FOR CALCULATION, BUT SHOW FULL EDGES LATER
            roi_edges = strict_region_of_interest(edges)

            lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 20, minLineLength=10, maxLineGap=50)
            
            left_line, right_line = average_slope_intercept(lines)

            if left_line is not None: left_lines_history.append(left_line)
            if right_line is not None: right_lines_history.append(right_line)

            left_avg = np.mean(left_lines_history, axis=0).astype(int) if left_lines_history else None
            right_avg = np.mean(right_lines_history, axis=0).astype(int) if right_lines_history else None

            # Calc Steering
            height, width = frame.shape[:2]
            offset, raw_angle = 0, 0
            if left_avg is not None and right_avg is not None:
                lane_center = (left_avg[0] + right_avg[0]) / 2
                offset = width / 2 - lane_center
                raw_angle = Kp * offset
            elif left_avg is not None:
                offset = width / 2 - (left_avg[0] + 300) 
                raw_angle = Kp * offset
            elif right_avg is not None:
                offset = width / 2 - (right_avg[0] - 300)
                raw_angle = Kp * offset
            
            smoothed_angle = smoothed_angle * (1 - ALPHA_STEER) + (90 + raw_angle) * ALPHA_STEER

            # --- YOLO ---
            detections = []
            results = yolo_objects(frame, verbose=False, conf=0.4)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    lbl = yolo_objects.names[int(box.cls[0])]
                    extra = {}
                    if 'traffic light' in lbl: extra['color'] = detect_traffic_light_color(frame[y1:y2, x1:x2])
                    detections.append((lbl, float(box.conf[0]), (x1,y1,x2,y2), extra))

            results_signs = yolo_signs(frame, verbose=False, conf=0.5)
            for r in results_signs:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if _cnn_model_cache is None: load_cnn_lazy()
                    lbl, conf, is_unk = classify_sign(_cnn_model_cache, frame[y1:y2, x1:x2])
                    detections.append((lbl, conf, (x1,y1,x2,y2), {'is_unknown': is_unk}))

            throttle, msg = calculate_throttle(detections)
            virtual_pwm = int(1500 + (throttle * 200))

            # --- DRAWING ---
            frame = draw_lane_lines(frame, left_avg, right_avg)
            for lbl, _, (x1, y1, x2, y2), extra in detections:
                col = (0, 0, 255) if 'stop' in lbl.lower() or extra.get('color')=='Rouge' else (0, 255, 0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                cv2.putText(frame, lbl, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            frame = draw_hud(frame, raw_angle, throttle, msg, virtual_pwm)

            # SHOW WINDOW 1
            cv2.imshow("Advanced Autonomous HUD", frame)
            
            # SHOW WINDOW 2: FULL EDGES (Showing "Everything" as requested)
            # We show 'edges' (full view) instead of 'roi_edges' (masked view)
            cv2.imshow("Lane Edges (Full View)", edges)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()