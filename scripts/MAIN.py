import h5py
import cv2
import os
import time
import numpy as np
from pathlib import Path
from argparse import Namespace
from rtmlib import RTMDet, RTMPose, draw_skeleton
from yolox.tracker.byte_tracker import BYTETracker
from paths import MODEL_DIR, DATA_DIR, OUTPUT_VIDEO_DIR, OUTPUT_H5_DIR, ensure_output_dirs

ensure_output_dirs()

#---------- CONFIGURATION ------------------
# Video Paths
record_output = False
IN_VIDEO_FILE = 'SAMPLE_17_01_2025_C2_S1.mp4'
OUT_VIDEO_FILE = 'test_det-M_pose-M_track-0508.mp4'
resize_output = False
resize_width = 960
resize_height = 540

# Data Paths
record_results = False
OUT_H5_FILE = "test_det-M_pose-M_track-0508.h5"

# Detection and tracking models
RTMDET_MODEL = 'rtmdet-m-640.onnx'
RTMPOSE_MODEL = 'rtmpose-m-256-192.onnx'

# RTMPose engine
device = 'cuda'
backend = 'onnxruntime'
#---------- CONFIGURATION ------------------

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
width = 1920
height = 1080
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
    track_buffer=60,
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
while cap.isOpened():

    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    cap_time = time.time()

    # Step 1: Detection
    det_bboxes, det_scores = detector(frame)  # [x1, y1, x2, y2, conf]
    det_time = time.time() 

    # Step 2: Format for ByteTrack
    if len(det_bboxes) > 0:
        dets_for_tracker = np.array([[*box, score, 0] for box, score in zip(det_bboxes, det_scores)]) if len(det_bboxes) > 0 else np.empty((0, 6)) # append dummy class_id=0
    else:
        dets_for_tracker = np.empty((0, 6))
    
    # Step 3: Tracking
    tracks = tracker.update(dets_for_tracker, [height, width], (height, width))
    track_time = time.time()

    # Step 4: Pose estimation (keypoints)
    img_show = frame.copy()    
    track_ids = []
    tracked_bboxes = []
    bbox_scores = [] # keep bbox scores
    bbox_rects = []  # (x1, y1, x2, y2) for drawing boxes and text    

    for track in tracks:
        x1, y1, w, h = track.tlwh
        x2, y2 = x1 + w, y1 + h
        track_id = int(track.track_id)

        track_ids.append(track_id)
        tracked_bboxes.append([x1, y1, x2, y2])
        bbox_scores.append(track.score if hasattr(track, "score") else 0.0)        
        bbox_rects.append((x1, y1, x2, y2, track_id, track.score if hasattr(track, "score") else None))

    if len(tracked_bboxes) > 0:
        keypoints_list, scores_list = pose_estimator(frame, tracked_bboxes)
    pose_time = time.time()

    # Build the HDF5 file
    if record_results:
        track_ids_array      = np.array(track_ids)
        bboxes_array         = np.array(tracked_bboxes)
        bbox_scores_array    = np.array(bbox_scores)
        keypoints_array      = np.array(keypoints_list)      # shape (N, K, 2)
        keypoint_scores_array = np.array(scores_list)        # shape (N, K)

        frame_group = h5file.create_group(f"frame_{frame_id:05d}")
        frame_group.create_dataset("track_ids", data=track_ids_array)
        frame_group.create_dataset("bboxes", data=bboxes_array)
        frame_group.create_dataset("bbox_scores", data=bbox_scores_array)
        frame_group.create_dataset("keypoints", data=keypoints_array)
        frame_group.create_dataset("keypoint_scores", data=keypoint_scores_array)

        # update trackID index
        for tid in track_ids:
            if tid not in track_id_index:
                track_id_index[tid] = []
            track_id_index[tid].append(frame_id)
    
    hdf_time = time.time()

    # ---DRAWING
    # Draw skeletons
    for keypoints, kpt_scores, track_id in zip(keypoints_list, scores_list, track_ids):
        img_show = draw_skeleton(
            img_show,
            np.array([keypoints]),        # shape (1, K, 2)
            np.array([kpt_scores]),       # shape (1, K)
            openpose_skeleton=False,
            kpt_thr=0.3,
            radius=3,
            line_width=2
        )

    # Draw bboxes and ID labels
    for (x1, y1, x2, y2, track_id, score) in bbox_rects:
        img_show = cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        label = f"ID: {track_id}"
        if score is not None:
            label += f" | {score:.2f}"

        img_show = cv2.putText(img_show, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    # Timing info
    disp_time = time.time()
    cap_duration = (cap_time - start_time) * 1000
    det_duration = (det_time - cap_time) * 1000
    track_duration = (track_time - det_time) * 1000
    pose_duration = (pose_time - track_time) * 1000
    hdf5_duration = (hdf_time - pose_time) * 1000
    disp_duration = (disp_time - hdf_time) * 1000
    

    img_show = cv2.putText(img_show, f'Volleyball Action Detection - FRANCOIS FRAYSSE @ UNISA', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 127, 0), 2)
    img_show = cv2.putText(img_show, f'cap: {cap_duration: .2f} ms', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'det: {det_duration: .2f} ms', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'track: {track_duration: .2f} ms', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'pose: {pose_duration: .2f} ms', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'hdf5: {hdf5_duration: .2f} ms', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show, f'disp: {disp_duration: .2f} ms', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
