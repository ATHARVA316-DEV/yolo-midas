#!/usr/bin/env python3
"""
midas_yolo_depth.py
Live depth estimation for specific objects using MiDaS and YOLOv8.
"""

import time
import cv2
import numpy as np
import onnxruntime as ort
import sys
from ultralytics import YOLO  # NEW: Import YOLO

# -----------------------
# Hardcoded configuration
# -----------------------
# MODEL_PATH = "models/midas_v21_384.onnx"
MODEL_PATH = "models/midas_v21_384.onnx"

# To this (example for a file in your Downloads folder):
MODEL_PATH = "C:\\Users\\mpath\\Downloads\\midas_v21_384.onnx"
VIDEO_SOURCE = 0        # 0 for default webcam
TARGET_FPS = 24.0
INVERT_DEPTH = False    # Set True if output looks inverted
USE_CUBIC_RESIZE = True

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# NEW: Depth thresholds for distance labels
NEAR_THRESHOLD = 0.7  # 0.0 (FAR) to 1.0 (NEAR). Tune this!
FAR_THRESHOLD = 0.2   # Tune this!

# -----------------------
# Helper Functions
# (choose_session, infer_model_input_size, preprocess, 
#  postprocess_depth, colorize_depth)
# ... NO CHANGES TO THESE FUNCTIONS ...
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

    # --- 2. NEW: Load YOLO (Object Detection Model) ---
    try:
        # 'yolov8n.pt' is the smallest, fastest model.
        # It will be downloaded automatically on first run.
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

            # --- 4. NEW: Run YOLO Object Detection ---
            # We run this on the original frame
            # verbose=False stops it from printing to console every frame
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

            # --- 6. NEW: Combine Results ---
            # Loop through all detected objects
            for r in yolo_results:
                for box in r.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    
                    # Get class name
                    cls_id = int(box.cls[0])
                    obj_label = yolo_model.names[cls_id]

                    # Get the depth region for this object
                    obj_depth_region = dmap[y1:y2, x1:x2]
                    
                    if obj_depth_region.size == 0:
                        continue # Skip if box is tiny or out of bounds
                    
                    # Get median depth (0.0=FAR, 1.0=NEAR)
                    median_depth = np.median(obj_depth_region)
                    
                    # Determine distance label
                    if median_depth > NEAR_THRESHOLD:
                        distance_label = "NEAR"
                        color = (0, 255, 0) # Green
                    elif median_depth < FAR_THRESHOLD:
                        distance_label = "FAR"
                        color = (0, 0, 255) # Red
                    else:
                        distance_label = "MID"
                        color = (0, 255, 255) # Yellow
                    
                    # --- 7. NEW: Draw Visualizations ---
                    
                    # Create combined label (e.g., "person: NEAR")
                    display_label = f"{obj_label}: {distance_label}"
                    
                    # Draw box on original frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, display_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                
                    # Draw box on depth map
                    cv2.rectangle(depth_viz, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(depth_viz, display_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # --- 8. Display Combined Output ---
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

            # --- 9. Handle FPS Cap and Quit Key ---
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