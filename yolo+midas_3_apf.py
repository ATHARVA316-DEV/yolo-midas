#!/usr/bin/env python3
"""
midas_yolo_depth.py
Live depth estimation, path planning, and contextual actions.

Uses MiDaS (depth), YOLOv8 (objects), and a simple
Artificial Potential Field (APF) for steering commands.

(Version 3: Corrected steering logic)
"""

import time
import cv2
import numpy as np
import onnxruntime as ort
import sys
from ultralytics import YOLO

# -----------------------
# Hardcoded configuration
# -----------------------
# MODEL_PATH = "models/midas_v21_384.onnx"
MODEL_PATH = "C:\\Users\\mpath\\Downloads\\midas_v21_384.onnx"
VIDEO_SOURCE = 0        # 0 for default webcam
TARGET_FPS = 24.0
INVERT_DEPTH = False    # Set True if output looks inverted
USE_CUBIC_RESIZE = True

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Depth thresholds for distance labels
NEAR_THRESHOLD = 0.7  # 0.0 (FAR) to 1.0 (NEAR). Tune this!
MID_THRESHOLD = 0.4   # Tune this!
FAR_THRESHOLD = 0.2   # Tune this!

# --- Path Planning Configuration ---
# This is the "safe zone" in the center of the screen.
SAFE_ZONE_WIDTH_RATIO = 0.4 

# This is the 'sensitivity' for steering. Higher = more sensitive.
STEER_THRESHOLD = 80.0 

# -----------------------
# Helper Functions
# (No changes to: choose_session, infer_model_input_size, 
#  preprocess, postprocess_depth, colorize_depth)
# -----------------------
def choose_session(model_path: str):
    available = ort.get_available_providers()
    windows_order = ['DmlExecutionProvider', 'CUDAExecutionProvider']
    chosen = []
    for p in windows_order:
        if p in available:
            chosen.append(p)
    if 'CPUExecutionProvider' in available:
        chosen.append('CPUExecutionProvider')
    if not chosen:
        print("No known providers found. Using default provider order.")
        sess = ort.InferenceSession(model_path)
    else:
        print("Available ONNX providers:", available)
        print("Using provider order:", chosen)
        sess = ort.InferenceSession(model_path, providers=chosen)
    return sess

def infer_model_input_size(sess):
    try:
        inp = sess.get_inputs()[0]
        shape = inp.shape
        dims = [d for d in shape if isinstance(d, int)]
        if len(dims) >= 2:
            h = dims[-2]
            w = dims[-1]
            if h == w:
                return int(h)
            else:
                return int(min(h, w))
    except Exception as e:
        print("Could not infer input shape from model metadata:", e)
    return 256

def preprocess(frame_bgr, target_size):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img_resized = (img_resized - MEAN) / STD
    img_chw = np.transpose(img_resized, (2, 0, 1)).astype(np.float32)
    return np.expand_dims(img_chw, 0)

def postprocess_depth(pred, out_w, out_h, invert=False):
    if pred is None:
        return np.zeros((out_h, out_w), dtype=np.float32)
    arr = pred
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    interp = cv2.INTER_CUBIC if USE_CUBIC_RESIZE else cv2.INTER_LINEAR
    arr_resized = cv2.resize(arr, (out_w, out_h), interpolation=interp)
    if invert:
        arr_resized = 1.0 / (arr_resized + 1e-6)
    dmin, dmax = float(arr_resized.min()), float(arr_resized.max())
    if dmax - dmin > 1e-6:
        norm = (arr_resized - dmin) / (dmax - dmin)
    else:
        norm = np.zeros_like(arr_resized)
    return norm

def colorize_depth(dmap_01):
    disp = (dmap_01 * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return color

# -----------------------
# Main loop
# -----------------------
def main():
    print("Starting MiDaS + YOLO object depth demo...")

    # --- 1. Load MiDaS (Depth Model) ---
    try:
        sess = choose_session(MODEL_PATH)
    except Exception as e:
        print(f"Failed to create ONNX session for MiDaS: {e}")
        sys.exit(1)
    target_size = infer_model_input_size(sess)
    print(f"MiDaS model input size: {target_size}")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # --- 2. Load YOLO (Object Detection Model) ---
    try:
        yolo_model = YOLO('yolov8n.pt')
        print("YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"Failed to load YOLOv8 model: {e}")
        print("Please ensure 'ultralytics' is installed: pip install ultralytics")
        sys.exit(1)

    # --- 3. Open Video Source ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {VIDEO_SOURCE}")
        sys.exit(1)

    min_frame_time = 1.0 / max(1.0, TARGET_FPS)
    print("Beginning capture. Press 'q' in the window to quit.")
    
    frame_count = 0
    start_t = time.time()
    fps_val = 0.0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            frame = cv2.flip(frame, 1)
            h0, w0 = frame.shape[:2]
            screen_center_x = w0 / 2
            
            # --- Define the "safe zone" for steering ---
            safe_zone_half_width = (w0 * SAFE_ZONE_WIDTH_RATIO) / 2
            safe_zone_left = screen_center_x - safe_zone_half_width
            safe_zone_right = screen_center_x + safe_zone_half_width
            
            # --- Draw the safe zone for visualization ---
            cv2.line(frame, (int(safe_zone_left), 0), (int(safe_zone_left), h0), (0, 255, 0), 1)
            cv2.line(frame, (int(safe_zone_right), 0), (int(safe_zone_right), h0), (0, 255, 0), 1)

            # --- 4. Run YOLO Object Detection ---
            yolo_results = yolo_model(frame, verbose=False)

            # --- 5. Run MiDaS Depth Estimation ---
            inp = preprocess(frame, target_size)
            try:
                out = sess.run([output_name], {input_name: inp})
                pred = out[0]
            except Exception as e:
                print(f"MiDaS inference error: {e}")
                break
                
            dmap = postprocess_depth(pred, w0, h0, invert=INVERT_DEPTH)
            depth_viz = colorize_depth(dmap)
            
            # --- Initialize navigation variables ---
            steering_force_x = 0.0  # Total horizontal "push" from all objects
            is_obstacle_ahead = False # Flag for immediate "STOP"

            # --- 6. Combine Results & Calculate Forces ---
            for r in yolo_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    cls_id = int(box.cls[0])
                    obj_label = yolo_model.names[cls_id]

                    obj_depth_region = dmap[y1:y2, x1:x2]
                    if obj_depth_region.size == 0: continue
                    
                    median_depth = np.median(obj_depth_region)
                    
                    # --- Determine distance label & color ---
                    if median_depth > NEAR_THRESHOLD:
                        distance_label = "NEAR"
                        color = (0, 0, 255) # Red
                    elif median_depth > MID_THRESHOLD:
                        distance_label = "MID"
                        color = (0, 255, 255) # Yellow
                    elif median_depth > FAR_THRESHOLD:
                        distance_label = "FAR"
                        color = (0, 255, 0) # Green
                    else:
                        distance_label = "V-FAR"
                        color = (255, 0, 0) # Blue
                    
                    # --- 7a. Contextual Action Logic ---
                    action_label = "" # Specific action for this object
                    is_obstacle = False # Does this object count for steering?

                    if obj_label == "dining table":
                        action_label = "Action: Go Around"
                        is_obstacle = True
                    elif obj_label == "door": # ⚠️ Will only work with a custom model!
                        action_label = "Action: Go to Door"
                        is_obstacle = False # We want to go TO it, not avoid it
                    elif obj_label == "person":
                        action_label = "Action: Wait"
                        is_obstacle = True # Treat people as obstacles
                    elif obj_label in ["chair", "sofa", "bed"]:
                        is_obstacle = True # Other things to avoid
                    
                    if is_obstacle and (distance_label == "NEAR" or distance_label == "MID"):
                        # --- 7b. Steering & Collision Logic ---
                        obj_center_x = (x1 + x2) / 2
                        
                        # --- Check for "STOP!" condition ---
                        is_in_safe_zone = (obj_center_x > safe_zone_left and obj_center_x < safe_zone_right)
                        if distance_label == "NEAR" and is_in_safe_zone:
                            is_obstacle_ahead = True # Trigger STOP!

                        # --- Calculate "push" force (APF) ---
                        # Strength is higher for nearer objects
                        strength = (median_depth ** 2) 
                        
                        # --- *** LOGIC FIX HERE *** ---
                        # 1. Calculate push direction:
                        #    Object on left -> positive push (to the right)
                        #    Object on right -> negative push (to the left)
                        push_direction = screen_center_x - obj_center_x
                        
                        # 2. Add this force to the total steering force
                        steering_force_x += (push_direction * strength)
                        # --- *** END OF FIX *** ---
                    
                    # --- 7c. Draw Visualizations ---
                    display_label = f"{obj_label}: {distance_label}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, display_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if action_label:
                        cv2.putText(frame, action_label, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                    cv2.rectangle(depth_viz, (x1, y1), (x2, y2), color, 2)

            # --- 8. Determine Final Navigation Command ---
            
            if is_obstacle_ahead:
                # Priority 1: STOP!
                nav_command = "STOP!"
                nav_color = (0, 0, 255) # Red
            elif steering_force_x > STEER_THRESHOLD:
                # Priority 2: Steer away from danger
                nav_command = "STEER RIGHT"
                nav_color = (0, 255, 255) # Yellow
            elif steering_force_x < -STEER_THRESHOLD:
                nav_command = "STEER LEFT"
                nav_color = (0, 255, 255) # Yellow
            else:
                # Priority 3: All clear
                nav_command = "GO STRAIGHT"
                nav_color = (0, 255, 0) # Green

            # --- Display the final navigation command ---
            cv2.putText(frame, nav_command, (w0 // 2 - 150, h0 - 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, nav_color, 3)
            
            # --- 9. Display Combined Output ---
            combined = np.concatenate((frame, depth_viz), axis=1)

            # FPS overlay
            frame_count += 1
            now = time.time()
            if now - start_t >= 1.0:
                fps_val = frame_count / (now - start_t)
                frame_count = 0
                start_t = now

            if fps_val > 0:
                cv2.putText(combined, f"FPS: {fps_val:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("RGB (L) | Depth (R)", combined)

            # --- 10. Handle FPS Cap and Quit Key ---
            elapsed = time.time() - t0
            sleep_time = min_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl-C)")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    main()