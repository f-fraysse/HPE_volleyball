import cv2
from functools import partial
from rtmlib import Custom, PoseTracker, RTMDet, RTMPose, draw_skeleton, draw_bbox
import time
import os
from sort import Sort
import numpy as np

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style, False for mmpose-style

RTMDET_MODEL = 'rtmdet-m-640.onnx'
RTMPOSE_MODEL = 'rtmpose-m-256-192.onnx'

RECORDING = False
in_file = 'SAMPLE_17_01_2025_C2_S1.mp4'
out_file = 'test_split_SORT_tuned.mp4'

#--------------------------------------------------
RTMDET_MODEL = os.path.join('D:\\PythonProjects\\HPE_volleyball\\models',RTMDET_MODEL)
RTMPOSE_MODEL = os.path.join('D:\\PythonProjects\\HPE_volleyball\\models',RTMPOSE_MODEL)


in_file = os.path.join('D:\\PythonProjects\\HPE_volleyball\\data', in_file)
out_file = os.path.join('D:\\PythonProjects\\HPE_volleyball\\output', out_file)
cap = cv2.VideoCapture(in_file)
width = 1920
height = 1080
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 60

if RECORDING:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'MJPG' or 'XVID' for AVI, 'mp4v' for MP4
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

detector = RTMDet(
            onnx_model=RTMDET_MODEL,
            model_input_size = (640, 640),
            backend=backend,
            device=device)

pose_estimator = RTMPose(
            onnx_model=RTMPOSE_MODEL,
            model_input_size = (192, 256),            
            backend=backend,
            device=device)

tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.2)  # Initialize SORT

while cap.isOpened():
    success, frame = cap.read() #all frames
    if not success:
        break
    
    # start detection and pose
    start_time = time.time()

    # Step 1: RTMDet for detection
    bboxes = detector(frame)
    det_time = time.time()

    # Step 2: Tracking
    if len(bboxes) == 0:
        tracked = np.empty((0, 5))
    else:
        tracked = tracker.update(np.array(bboxes))  # shape: [N, 5] → [x1, y1, x2, y2, ID]
    track_time = time.time()

    # Step 3: RTMPose for Pose Estimation on Tracked Boxes
    tracked_bboxes = [t[:4] for t in tracked]
    track_ids = [int(t[4]) for t in tracked]
    keypoints, scores = pose_estimator(frame, tracked_bboxes)
    pose_time = time.time()

    det_dur = (det_time - start_time) * 1000 #ms
    track_dur = (track_time - det_time) * 1000 #ms
    pose_dur = (pose_time - track_time) * 1000 #ms

    img_show = frame.copy()

    for kpt, score, track_id in zip(keypoints, scores, track_ids):
        img_show = draw_skeleton(
            img_show,
            np.array([kpt]),
            np.array([score]),
            openpose_skeleton=openpose_skeleton,
            kpt_thr=0.3,
            line_width=3
        )
        x, y = int(kpt[0][0]), int(kpt[0][1])
        img_show = cv2.putText(img_show, f'{track_id}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    img_show = draw_bbox(img_show, bboxes, (255, 0, 0))    
    # img_show = cv2.resize(img_show, (960, 640))

    img_show = cv2.putText(img_show,f'det: {det_dur: .2f} ms',(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show,f'track: {track_dur: .2f} ms',(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_show = cv2.putText(img_show,f'pose: {pose_dur: .2f} ms',(10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Body and Feet Pose Estimation', img_show)

    if RECORDING:
        out.write(img_show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Press 'q' to exit
        break

cap.release()
if RECORDING:
    out.release()
cv2.destroyAllWindows()
