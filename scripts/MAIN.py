import h5py
import cv2
import os
import time
# from time import perf_counter
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from argparse import Namespace
from rtmlib import RTMDet, RTMPose, YOLOX, draw_skeleton
from yolox.tracker.byte_tracker import BYTETracker
from paths import MODEL_DIR, DATA_DIR, OUTPUT_VIDEO_DIR, OUTPUT_H5_DIR, ensure_output_dirs

ensure_output_dirs()

#---------- CONFIGURATION ------------------
# Video Paths
record_output = False
IN_VIDEO_FILE = 'SAMPLE_17_01_2025_C2_S1.mp4'
# Reset output filename to avoid confusion with interval tests
OUT_VIDEO_FILE = 'SAMPLE_det-M_pose-M_track-EveryFrame.mp4'
resize_output = False
resize_width = 960
resize_height = 540

# Data Paths
record_results = False
OUT_H5_FILE = "SAMPLE2_det-M_pose-M_track-EveryFrame.h5"

# Detection and tracking models
RTMDET_MODEL = 'rtmdet-m-640.onnx'
RTMPOSE_MODEL = 'rtmpose-m-256-192_26k.onnx'

# RTMPose engine
device = 'cuda'
backend = 'onnxruntime'
#---------- CONFIGURATION ------------------

# Create profiling logs directory
log_dir = "profiling_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"profiling_{timestamp}_EveryFrame.csv") # Add suffix to log file

# Initialize CSV log file with headers
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'frame_id',
        'det_total', 'det_preprocess', 'det_prep', 'det_model', 'det_postprocess',
        'pose_total', 'pose_preprocess', 'pose_prep', 'pose_model', 'pose_postprocess', 'pose_num_bboxes',
        'cap_time', 'track_time', 'hdf5_time', 'disp_time', 'total_frame_time'
    ])

# Add these variables to track timing statistics
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

# Make the full path + file names
RTMDET_MODEL = os.path.join(MODEL_DIR, RTMDET_MODEL)
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

# Init detector
detector = RTMDet(
    onnx_model=RTMDET_MODEL,
    model_input_size=(640, 640),
    backend=backend,
    device=device
)

# Init ByteTrack tracker
args = Namespace(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=60, # Keep original buffer setting
    frame_rate=fps,
    mot20=False,
    min_hits=3
)
tracker = BYTETracker(args)

# init pose detector
pose_estimator = RTMPose(
            onnx_model=RTMPOSE_MODEL,
            model_input_size = (192, 256),
            backend=backend,
            device=device)

# ------------ START LOOP OVER FRAMES --------------
frame_id = 0
global_start = time.time()

while cap.isOpened():

    start_time = time.perf_counter()
    success, frame = cap.read()
    if not success:
        break
    frame_id += 1
    cap_time = time.perf_counter()

    # Step 1: Detection (runs every frame)
    det_bboxes_scores, det_timing = detector(frame)  # [x1, y1, x2, y2, conf]
    det_bboxes, det_scores = det_bboxes_scores
    det_time = time.perf_counter()

    # Update detection timing statistics
    for key in det_timing:
        if key in det_timing_stats:
            det_timing_stats[key].append(det_timing[key])

    # Step 2: Format for ByteTrack
    if len(det_bboxes) > 0:
        dets_for_tracker = np.array([[*box, score, 0] for box, score in zip(det_bboxes, det_scores)])
    else:
        dets_for_tracker = np.empty((0, 6))

    # Step 3: Tracking
    tracks = tracker.update(dets_for_tracker, [height, width], (height, width))
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
        keypoints_list, scores_list, pose_timing = pose_estimator(frame, tracked_bboxes)
        # Update pose timing statistics
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

    # Draw bboxes and ID labels (using bbox_rects from Step 4)
    for (x1, y1, x2, y2, track_id, score) in bbox_rects:
        img_show = cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) # Blue boxes

        label = f"ID: {track_id}"
        if score is not None:
            label += f" | {score:.2f}"

        img_show = cv2.putText(img_show, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # Blue text

    # Timing info
    disp_time = time.perf_counter()
    cap_duration = (cap_time - start_time) * 1000
    det_duration = (det_time - cap_time) * 1000
    track_duration = (track_time - det_time) * 1000
    pose_duration = (pose_time - track_time) * 1000
    hdf5_duration = (hdf_time - pose_time) * 1000
    disp_duration = (disp_time - hdf_time) * 1000
    total_frame_time = (disp_time - start_time) * 1000

    # Write to CSV log
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_id,
            det_timing['total'], det_timing['preprocess'], det_timing['prep'], det_timing['model'], det_timing['postprocess'],
            pose_timing['total'], pose_timing['preprocess'], pose_timing['prep'], pose_timing['model'], pose_timing['postprocess'],
            pose_timing['num_bboxes'],
            cap_duration, track_duration, hdf5_duration, disp_duration, total_frame_time
        ])


    img_show = cv2.putText(img_show, f'Volleyball Action Detection - FRANCOIS FRAYSSE @ UNISA', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 127, 0), 2)
    img_show = cv2.putText(img_show, f'cap: {cap_duration: .1f} ms', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'det: {det_duration: .1f} ms', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'track: {track_duration: .1f} ms', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'pose: {pose_duration: .1f} ms', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'hdf5: {hdf5_duration: .1f} ms', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'disp: {disp_duration: .1f} ms', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize & show
    if resize_output:
        img_show = cv2.resize(img_show, (resize_width, resize_height))
    cv2.imshow('Tracking Output', img_show)

    if record_output:
        out.write(img_show)

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

# Print summary statistics (Reverted to original format)
print("\n===== DETECTION TIMING STATISTICS =====")
for key in det_timing_stats:
    times = det_timing_stats[key][1:] # Skip first frame if needed for stable stats
    if times:
        print(f"{key}: min={min(times):.2f}ms, max={max(times):.2f}ms, avg={sum(times)/len(times):.2f}ms, median={sorted(times)[len(times)//2]:.2f}ms")

print("\n===== POSE ESTIMATION TIMING STATISTICS =====")
for key in pose_timing_stats:
    times = pose_timing_stats[key][1:] # Skip first frame if needed
    if times:
        print(f"{key}: min={min(times):.2f}ms, max={max(times):.2f}ms, avg={sum(times)/len(times):.2f}ms, median={sorted(times)[len(times)//2]:.2f}ms")

print(f"\nDetailed profiling data saved to: {log_file}")
