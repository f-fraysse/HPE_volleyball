# You can run this simple script to test RTMLIB is installed and working (no tracking)

import cv2
from rtmlib import Body, draw_skeleton, draw_bbox
import time
import os

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style, False for mmpose-style

RECORDING = True
in_file = 'SAMPLE_17_01_2025_C2_S1.mp4'
out_file = 'test.mp4'

in_file = os.path.join('D:\\PythonProjects\\HPE_volleyball\\data', in_file)
out_file = os.path.join('D:\\PythonProjects\\HPE_volleyball\\output', out_file)
cap = cv2.VideoCapture(in_file)
width =960 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = 640 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 60

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'MJPG' or 'XVID' for AVI, 'mp4v' for MP4
if RECORDING:
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

body = Body(
        mode='performance',
        # pose='rtmo',
        to_openpose=openpose_skeleton,
        backend=backend,
        device=device)

while cap.isOpened():
    success, frame = cap.read() #all frames
    if not success:
        break

    start_time = time.time()
    keypoints, scores, bboxes = body(frame)
    det_time = (time.time() - start_time) * 1000 #ms

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.2,
                             line_width=3)
    img_show = draw_bbox(img_show, bboxes, (255, 0, 0))
    
    img_show = cv2.resize(img_show, (960, 640))
    img_show = cv2.putText(img_show,f'det: {det_time: .2f} ms',(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Body and Feet Pose Estimation', img_show)

    if RECORDING:
        out.write(img_show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Press 'q' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
