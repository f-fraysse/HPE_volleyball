import cv2
import os
import time
import numpy as np
import statistics
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from paths import MODEL_DIR, DATA_DIR, OUTPUT_VIDEO_DIR, ensure_output_dirs

# Ensure output directories exist
ensure_output_dirs()

#---------- CONFIGURATION ------------------
# Video Paths
record_output = True
IN_VIDEO_FILE = 'SAMPLE_17_01_2025_C2_S1.mp4'
OUT_VIDEO_FILE = 'SAMPLE_YOLO11_TensorRT.mp4'
resize_output = False
resize_width = 960
resize_height = 540

# Model parameters
det_conf_threshold = 0.5
pose_conf_threshold = 0.5
tracker_type = "bytetrack.yaml"
#---------- CONFIGURATION ------------------

# Make the full path + file names
IN_VIDEO_FILE = os.path.join(DATA_DIR, IN_VIDEO_FILE)
OUT_VIDEO_FILE = os.path.join(OUTPUT_VIDEO_DIR, OUT_VIDEO_FILE)

# Create profiling logs directory
log_dir = "profiling_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"profiling_YOLO11_TensorRT_{timestamp}.csv")

# Load video
cap = cv2.VideoCapture(IN_VIDEO_FILE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Init output video writer
if record_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if resize_output:
        out = cv2.VideoWriter(OUT_VIDEO_FILE, fourcc, fps, (resize_width, resize_height))
    else:
        out = cv2.VideoWriter(OUT_VIDEO_FILE, fourcc, fps, (width, height))

# Initialize detection model with TensorRT
print("Loading TensorRT detection model...")
detection_model = YOLO('yolo11m.engine')  # TensorRT engine
detection_model.conf = det_conf_threshold

# Initialize pose model with TensorRT
print("Loading TensorRT pose model...")
pose_model = YOLO('yolo11m-pose.engine')  # TensorRT engine
pose_model.conf = pose_conf_threshold

# Timing variables
cap_times = []
det_times = []
track_times = []
pose_times = []
total_times = []

# ------------ START LOOP OVER FRAMES --------------
frame_id = 0
global_start = time.time()

print("Starting video processing...")
while cap.isOpened():
    # Read frame
    start_time = time.perf_counter()
    success, frame = cap.read()
    if not success:
        break
    frame_id += 1
    cap_time = time.perf_counter()
    
    # Detection + Tracking (combined to avoid running detection twice)
    track_results = detection_model.track(frame, persist=True, tracker=tracker_type, classes=0, verbose=False)
    det_track_time = time.perf_counter()
    
    # Pose estimation on the whole frame
    pose_results = None
    if track_results[0].boxes.id is not None and len(track_results[0].boxes.id) > 0:
        # Run pose estimation on the entire frame
        pose_results = pose_model(frame, verbose=False)
    
    pose_time = time.perf_counter()
    
    # Visualization
    img_show = frame.copy()
    
    # Draw tracked boxes and IDs
    if track_results[0].boxes.id is not None:
        tracked_boxes = track_results[0].boxes.xyxy.cpu().numpy()
        track_ids = track_results[0].boxes.id.int().cpu().numpy()
        
        for i, box in enumerate(tracked_boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = track_ids[i]
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_show, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw pose keypoints
    if pose_results is not None and len(pose_results[0].keypoints) > 0:
        # Get the original image with keypoints drawn
        pose_img = pose_results[0].plot()
        
        # Overlay pose keypoints on our image with bounding boxes
        alpha = 0.5
        img_show = cv2.addWeighted(img_show, 1 - alpha, pose_img, alpha, 0)
    
    # Calculate timing information
    cap_duration = (cap_time - start_time) * 1000
    det_track_duration = (det_track_time - cap_time) * 1000
    pose_duration = (pose_time - det_track_time) * 1000
    total_duration = (pose_time - start_time) * 1000
    
    # Store timing data (excluding first frame)
    if frame_id > 1:
        cap_times.append(cap_duration)
        det_times.append(det_track_duration)  # Combined detection and tracking
        pose_times.append(pose_duration)
        total_times.append(total_duration)
    
    # Display timing information
    img_show = cv2.putText(img_show, f'Volleyball YOLO11 TensorRT Pipeline', (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 127, 0), 2)
    img_show = cv2.putText(img_show, f'cap: {cap_duration:.1f} ms', (10, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'det+track: {det_track_duration:.1f} ms', (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'pose: {pose_duration:.1f} ms', (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    fps = 1000 / total_duration if total_duration > 0 else 0
    img_show = cv2.putText(img_show, f'total: {total_duration:.1f} ms ({fps:.1f} FPS)', 
                          (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Resize if needed
    if resize_output:
        img_show = cv2.resize(img_show, (resize_width, resize_height))
    
    # Display and write output
    cv2.imshow('YOLO11 TensorRT Tracking', img_show)
    if record_output:
        out.write(img_show)
    
    # Print progress every 10 frames
    if frame_id % 10 == 0:
        print(f"Processed frame {frame_id}, current FPS: {fps:.1f}")
    
    # Check for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
if record_output:
    out.release()
cv2.destroyAllWindows()

finish_time = time.time()
print(f"Total processing time: {(finish_time - global_start):.1f} seconds for {frame_id} frames")

# Calculate and print performance statistics
def calculate_stats(times_list):
    if not times_list:
        return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
    return {
        'min': min(times_list),
        'max': max(times_list),
        'avg': sum(times_list) / len(times_list),
        'median': statistics.median(times_list)
    }

# Print performance summary
print("\n===== PERFORMANCE SUMMARY =====")
print(f"{'Component':<10}\t{'Min (ms)':<10}\t{'Max (ms)':<10}\t{'Avg (ms)':<10}\t{'Median (ms)':<10}")
print("-" * 60)

stats = {
    "Capture": calculate_stats(cap_times),
    "Det+Track": calculate_stats(det_times),
    "Pose": calculate_stats(pose_times),
    "Total": calculate_stats(total_times)
}

for name, stat in stats.items():
    print(f"{name:<10}\t{stat['min']:<10.1f}\t{stat['max']:<10.1f}\t{stat['avg']:<10.1f}\t{stat['median']:<10.1f}")

# Calculate overall FPS
avg_total = stats["Total"]["avg"]
avg_fps = 1000 / avg_total if avg_total > 0 else 0
print(f"\nAverage FPS: {avg_fps:.1f}")
