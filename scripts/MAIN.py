import h5py
import cv2
import os
import sys
import time
import numpy as np
import csv
import statistics # For median calculation
from datetime import datetime
from pathlib import Path
from argparse import Namespace
from rtmlib import RTMDet, RTMPose, YOLOX, draw_skeleton
from yolox.tracker.byte_tracker import BYTETracker
from paths import MODEL_DIR, DATA_DIR, OUTPUT_VIDEO_DIR, OUTPUT_H5_DIR, ensure_output_dirs

# Add project root to path for pipeline imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pipeline components
from pipeline.detector_base import create_detector
from pipeline.tracker_adapter import create_tracker, update_tracker
from pipeline.pose_adapter import create_pose_estimator, estimate_pose

ensure_output_dirs()

#---------- CONFIGURATION ------------------
# Video Paths
record_output = True
IN_VIDEO_FILE = 'SAMPLE_17_01_2025_C2_S1.mp4'
# Reset output filename to avoid confusion with interval tests
OUT_VIDEO_FILE = 'SAMPLE_17_01_2025_C2_S1_detr-l_pose-M.mp4'
resize_output = False
resize_width = 960
resize_height = 540

# Data Paths
record_results = False
OUT_H5_FILE = "SAMPLE_LONG_det-M_pose-M_track-EveryFrame.h5"

# Profiling
record_profiling = True

# Detection and tracking models
DETECTOR = 'yolox'  # Options: 'rtmdet', 'rtdetr', 'yolox'
DISPLAY_BALL_DETECTIONS = False  # Only applies when using RT-DETR or YOLOX
RTMDET_MODEL = 'rtmdet-m-640.onnx'
RTDETR_MODEL = 'rtdetrv2_r50vd_640.onnx'  # large
YOLOX_MODEL = 'yolox_l.onnx'
RTMPOSE_MODEL = 'rtmpose-m-256-192.onnx'

# RTMPose engine
device = 'cuda'
backend = 'onnxruntime'
#---------- CONFIGURATION ------------------

# Create profiling logs directory and initialize profiling variables
if record_profiling:
    log_dir = "profiling_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"profiling_{timestamp}_EveryFrame.csv") # Add suffix to log file

    # Initialize CSV log file with headers
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_id',
            'det_total', 'det_preprocess', 'det_prep', 'det_model', 'det_postprocess', # Detailed detection
            'pose_total', 'pose_preprocess', 'pose_prep', 'pose_model', 'pose_postprocess', 'pose_num_bboxes', # Detailed pose
            'cap_time_ms', 'det_time_ms', 'track_time_ms', 'pose_time_ms', 'hdf5_time_ms', # Overall components
            'disp_time_ms', 'csv_time_ms', 'draw_time_ms', 'total_frame_time_ms' # Overall components cont.
        ])

    # Add these variables to track DETAILED timing statistics
    det_timing_stats = {
        'total': [],
        'preprocess': [],
        'prep': [],
        'model': [],
        'postprocess': []
    }

    pose_timing_stats = {
        'total': [],
        'preprocess': [],
        'prep': [],
        'model': [],
        'postprocess': []
    }

    # Add lists to store OVERALL timing durations for final stats
    cap_times_ms = []
    det_times_ms = []
    track_times_ms = []
    pose_times_ms = []
    hdf_times_ms = []
    disp_times_ms = []
    csv_times_ms = []
    draw_times_ms = []
    total_frame_times_ms = []


# Make the full path + file names
if DETECTOR == 'rtmdet':
    DETECTOR_MODEL = os.path.join(MODEL_DIR, RTMDET_MODEL)
elif DETECTOR == 'rtdetr':
    DETECTOR_MODEL = os.path.join(MODEL_DIR, RTDETR_MODEL)
elif DETECTOR == 'yolox':
    DETECTOR_MODEL = os.path.join(MODEL_DIR, YOLOX_MODEL)
else:
    raise ValueError(f"Unsupported detector: {DETECTOR}")

RTMPOSE_MODEL = os.path.join(MODEL_DIR, RTMPOSE_MODEL)
IN_VIDEO_FILE = os.path.join(DATA_DIR, IN_VIDEO_FILE)
OUT_VIDEO_FILE = os.path.join(OUTPUT_VIDEO_DIR, OUT_VIDEO_FILE)
OUT_H5_FILE = os.path.join(OUTPUT_H5_DIR, OUT_H5_FILE)

# create results HDF5 file
if record_results:
    h5file = h5py.File(OUT_H5_FILE, "w")
track_id_index = {}  # Will be populated frame-by-frame

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

# Init detector using factory
detector = create_detector(DETECTOR, DETECTOR_MODEL, device, backend, ball_conf_threshold=0.5)

# Init ByteTrack tracker using adapter
tracker = create_tracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=fps)

# Init pose detector using adapter
pose_estimator = create_pose_estimator(RTMPOSE_MODEL, device, backend)

# ------------ START LOOP OVER FRAMES --------------
frame_id = 0
global_start = time.time()
# Initialize durations from previous frame for the first iteration's total calculation and display
csv_duration_ms = 0.0
draw_duration_ms = 0.0

# Create a named window with normal flags (needed for full screen toggle)
cv2.namedWindow("FullScreen", cv2.WINDOW_NORMAL)

# Set the window property to full screen
cv2.setWindowProperty("FullScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():

    start_time = time.perf_counter()
    success, frame = cap.read()
    if not success:
        break
    frame_id += 1
    cap_time = time.perf_counter()

    # Step 1: Detection (runs every frame)
    det_results, det_timing = detector(frame)
    
    # Handle different detector return formats
    if len(det_results) == 3:
        # RT-DETR returns (bboxes, scores, class_labels)
        det_bboxes, det_scores, det_classes = det_results
    else:
        # RTMDet returns (bboxes, scores) - all are persons
        det_bboxes, det_scores = det_results
        det_classes = np.zeros(len(det_bboxes), dtype=np.int32)  # All class 0 (person)
    
    det_time = time.perf_counter()

    # Update detection timing statistics
    if record_profiling:
        for key in det_timing:
            if key in det_timing_stats:
                det_timing_stats[key].append(det_timing[key])

    # Step 2: Separate person and ball detections
    person_mask = det_classes == 0
    ball_mask = det_classes == 32
    
    person_bboxes = det_bboxes[person_mask] if person_mask.any() else np.empty((0, 4))
    person_scores = det_scores[person_mask] if person_mask.any() else np.empty((0,))
    
    ball_bboxes = det_bboxes[ball_mask] if ball_mask.any() else np.empty((0, 4))
    ball_scores = det_scores[ball_mask] if ball_mask.any() else np.empty((0,))
    
    # Format person detections for ByteTrack (only track persons)
    if len(person_bboxes) > 0:
        dets_for_tracker = np.array([[*box, score, 0] for box, score in zip(person_bboxes, person_scores)])
    else:
        dets_for_tracker = np.empty((0, 6))

    # Step 3: Tracking
    tracks = update_tracker(tracker, dets_for_tracker, [height, width], (height, width))
    track_time = time.perf_counter()

    # Step 4: Prepare data for Pose Estimation and Drawing (directly from tracker output)
    img_show = frame.copy()
    track_ids = []
    tracked_bboxes = [] # BBoxes for pose estimator input
    bbox_scores = []    # Scores corresponding to tracked_bboxes
    bbox_rects = []     # Data for drawing boxes/labels

    for track in tracks:
        # Only process tracks that are currently active/tracked
        if not track.is_activated:
             continue # Skip lost tracks for pose/drawing

        x1, y1, w, h = track.tlwh
        x2, y2 = x1 + w, y1 + h
        track_id = int(track.track_id)
        score = track.score if hasattr(track, "score") else 0.0

        track_ids.append(track_id)
        tracked_bboxes.append([x1, y1, x2, y2])
        bbox_scores.append(score)
        bbox_rects.append((x1, y1, x2, y2, track_id, score))

    # Step 5: Pose estimation (keypoints)
    # Initialize pose timing info
    pose_timing = {
        'total': 0, 'preprocess': 0, 'prep': 0, 'model': 0, 'postprocess': 0, 'num_bboxes': 0
    }
    keypoints_list = [] # Ensure these are initialized
    scores_list = []

    if len(tracked_bboxes) > 0:
        keypoints_list, scores_list, pose_timing = estimate_pose(pose_estimator, frame, tracked_bboxes)
        # Update pose timing statistics
        if record_profiling:
            for key in pose_timing:
                if key in pose_timing_stats and key != 'num_bboxes':
                    pose_timing_stats[key].append(pose_timing[key])
    # else: keypoints_list, scores_list remain empty

    pose_time = time.perf_counter()

    # Step 6: Build the HDF5 file
    if record_results:
        # Ensure data corresponds to the tracks processed in this frame
        track_ids_array = np.array(track_ids)
        bboxes_array = np.array(tracked_bboxes) # Should be xyxy if needed, check format
        # Convert tlwh from bbox_rects to xyxy if needed for HDF5 consistency
        # bboxes_array = np.array([[r[0], r[1], r[2], r[3]] for r in bbox_rects]) # Example if xyxy needed
        bbox_scores_array = np.array(bbox_scores)
        keypoints_array = np.array(keypoints_list)      # shape (N, K, 2)
        keypoint_scores_array = np.array(scores_list)   # shape (N, K)

        if track_ids_array.size > 0: # Only save if there's valid data
            frame_group = h5file.create_group(f"frame_{frame_id:05d}")
            frame_group.create_dataset("track_ids", data=track_ids_array)
            frame_group.create_dataset("bboxes", data=bboxes_array) # Save the bboxes used for pose
            frame_group.create_dataset("bbox_scores", data=bbox_scores_array)
            frame_group.create_dataset("keypoints", data=keypoints_array)
            frame_group.create_dataset("keypoint_scores", data=keypoint_scores_array)

            # update trackID index
            for tid in track_ids: # Use track_ids from this frame
                if tid not in track_id_index:
                    track_id_index[tid] = []
                track_id_index[tid].append(frame_id)

    hdf_time = time.perf_counter()

    # ---DRAWING---
    # Draw skeletons (matched by order with track_ids)
    for keypoints, kpt_scores in zip(keypoints_list, scores_list):
         img_show = draw_skeleton(
            img_show,
            np.array([keypoints]),        # shape (1, K, 2)
            np.array([kpt_scores]),       # shape (1, K)
            openpose_skeleton=False,
            kpt_thr=0.3,
            radius=3,
            line_width=2
        )

    # Draw person bboxes and ID labels (using bbox_rects from Step 4)
    for (x1, y1, x2, y2, track_id, score) in bbox_rects:
        img_show = cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) # Blue boxes

        label = f"ID: {track_id}"
        if score is not None:
            label += f" | {score:.2f}"

        img_show = cv2.putText(img_show, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # Blue text
    
    # Draw ball bboxes in RED (only if enabled)
    if DISPLAY_BALL_DETECTIONS:
        for (x1, y1, x2, y2), score in zip(ball_bboxes, ball_scores):
            img_show = cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red boxes
            label = f"Ball | {score:.2f}"
            img_show = cv2.putText(img_show, label, (int(x1), int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red text

    # Timing info calculation
    disp_time = time.perf_counter() # End of drawing skeletons/bboxes

    cap_duration = (cap_time - start_time) * 1000
    det_duration = (det_time - cap_time) * 1000
    track_duration = (track_time - det_time) * 1000
    pose_duration = (pose_time - track_time) * 1000
    hdf5_duration = (hdf_time - pose_time) * 1000
    disp_duration = (disp_time - hdf_time) * 1000 # Time for drawing skeletons/bboxes

    # --- CSV Write Timing ---
    csv_write_start_time = time.perf_counter()
    # Write to CSV log (using previous frame's csv/draw times for total calculation consistency)
    # Note: total_frame_time uses csv_duration_ms and draw_duration_ms from the *previous* frame
    total_frame_time = cap_duration + det_duration + track_duration + pose_duration + hdf5_duration + disp_duration + csv_duration_ms + draw_duration_ms
    if record_profiling:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                # Detailed timings (from detector/pose estimator internal profiling)
                det_timing.get('total', 0), det_timing.get('preprocess', 0), det_timing.get('prep', 0), det_timing.get('model', 0), det_timing.get('postprocess', 0),
                pose_timing.get('total', 0), pose_timing.get('preprocess', 0), pose_timing.get('prep', 0), pose_timing.get('model', 0), pose_timing.get('postprocess', 0),
                pose_timing.get('num_bboxes', 0),
                # Overall component timings (calculated in this script)
                cap_duration, det_duration, track_duration, pose_duration, hdf5_duration,
                disp_duration, csv_duration_ms, draw_duration_ms, total_frame_time # Use previous frame's csv/draw
            ])
    csv_time = time.perf_counter() # End of CSV write
    current_csv_duration = (csv_time - csv_write_start_time) * 1000 # CSV write time for *this* frame

    # --- Final Drawing and Display Timing ---
    # Draw timing text overlays (using previous frame's csv/draw times)
    img_show = cv2.putText(img_show, f'Volleyball Action Detection - FRANCOIS FRAYSSE @ UNISA', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 127, 0), 2)
    img_show = cv2.putText(img_show, f'cap: {cap_duration:.1f} ms', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'det: {det_duration:.1f} ms', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'track: {track_duration:.1f} ms', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'pose: {pose_duration:.1f} ms', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'hdf5: {hdf5_duration:.1f} ms', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'disp: {disp_duration:.1f} ms', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Drawing bboxes/skeletons
    img_show = cv2.putText(img_show, f'csv: {csv_duration_ms:.1f} ms', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Previous frame's csv write
    img_show = cv2.putText(img_show, f'draw: {draw_duration_ms:.1f} ms', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Previous frame's draw/display
    fps_display = 1000 / total_frame_time if total_frame_time > 0 else 0
    img_show = cv2.putText(img_show, f'total: {total_frame_time:.1f} ms ({fps_display:.0f} FPS)', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize & show
    if resize_output:
        img_show = cv2.resize(img_show, (resize_width, resize_height))
    cv2.imshow('FullScreen', img_show)

    if record_output:
        out.write(img_show)

    draw_time = time.perf_counter() # End of drawing text/displaying/writing frame
    current_draw_duration = (draw_time - csv_time) * 1000 # Draw time for *this* frame

    # Store overall timings for final stats
    if record_profiling and frame_id > 1: # Optionally skip first frame for more stable stats
        cap_times_ms.append(cap_duration)
        det_times_ms.append(det_duration)
        track_times_ms.append(track_duration)
        pose_times_ms.append(pose_duration)
        hdf_times_ms.append(hdf5_duration)
        disp_times_ms.append(disp_duration)
        csv_times_ms.append(current_csv_duration) # Store this frame's csv time
        draw_times_ms.append(current_draw_duration) # Store this frame's draw time
        total_frame_times_ms.append(total_frame_time) # Store total calculated with previous csv/draw

    # Update durations for the next frame's calculation/display
    csv_duration_ms = current_csv_duration
    draw_duration_ms = current_draw_duration

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
if record_output:
    out.release()
cv2.destroyAllWindows()

# Save track presence info to HDF5
if record_results:
    index_group = h5file.create_group("track_presence")
    for tid, frames in track_id_index.items():
        index_group.create_dataset(str(tid), data=np.array(frames, dtype='int32'))

    h5file.close()

finish_time = time.time()
print(f"total time: {(finish_time - global_start):.1f} seconds")

if record_profiling:
    # Helper function to calculate statistics
    def calculate_stats(times_list):
        if not times_list:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        return {
            'min': min(times_list),
            'max': max(times_list),
            'avg': sum(times_list) / len(times_list),
            'median': statistics.median(times_list)
        }

    # Calculate overall statistics
    overall_stats = {
        "Total": calculate_stats(total_frame_times_ms),
        "Capture": calculate_stats(cap_times_ms),
        "Detection": calculate_stats(det_times_ms),
        "Track": calculate_stats(track_times_ms),
        "Pose": calculate_stats(pose_times_ms),
        "Hdf": calculate_stats(hdf_times_ms),
        "Disp": calculate_stats(disp_times_ms),
        "Csv": calculate_stats(csv_times_ms),
        "Draw": calculate_stats(draw_times_ms),
    }

    # Print OVERALL summary statistics (Tab Aligned)
    print("\n===== OVERALL TIMING STATISTICS =====")
    # Print header
    print(f"{'Component':<10}\t{'Min (ms)':<10}\t{'Max (ms)':<10}\t{'Avg (ms)':<10}\t{'Median (ms)':<10}")
    print("-" * 60) # Separator line
    for name, stats in overall_stats.items():
        print(f"{name:<10}\t{stats['min']:<10.1f}\t{stats['max']:<10.1f}\t{stats['avg']:<10.1f}\t{stats['median']:<10.1f}")


    # Print DETAILED summary statistics (Tab Aligned)
    print("\n===== DETECTION TIMING STATISTICS =====")
    print(f"{'Component':<12}\t{'Min (ms)':<10}\t{'Max (ms)':<10}\t{'Avg (ms)':<10}\t{'Median (ms)':<10}")
    print("-" * 65) # Separator line
    for key in det_timing_stats:
        # Use times collected by RTMDet/RTMPose directly, skip first frame for stable stats
        times = det_timing_stats[key][1:]
        if times:
            # Calculate median using statistics library for consistency
            min_val = min(times)
            max_val = max(times)
            avg_val = sum(times)/len(times)
            median_val = statistics.median(times) if len(times) > 0 else 0
            print(f"{key:<12}\t{min_val:<10.1f}\t{max_val:<10.1f}\t{avg_val:<10.1f}\t{median_val:<10.1f}")

    print("\n===== POSE ESTIMATION TIMING STATISTICS =====")
    print(f"{'Component':<12}\t{'Min (ms)':<10}\t{'Max (ms)':<10}\t{'Avg (ms)':<10}\t{'Median (ms)':<10}")
    print("-" * 65) # Separator line
    for key in pose_timing_stats:
        # Use times collected by RTMDet/RTMPose directly, skip first frame for stable stats
        times = pose_timing_stats[key][1:]
        if times:
            min_val = min(times)
            max_val = max(times)
            avg_val = sum(times)/len(times)
            median_val = statistics.median(times) if len(times) > 0 else 0
            print(f"{key:<12}\t{min_val:<10.1f}\t{max_val:<10.1f}\t{avg_val:<10.1f}\t{median_val:<10.1f}")

    print(f"\nDetailed profiling data saved to: {log_file}")
