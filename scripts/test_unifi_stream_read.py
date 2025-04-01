import cv2
import os
import time
import numpy as np
from rtmlib import RTMDet, RTMPose, draw_skeleton
from paths import MODEL_DIR, DATA_DIR, OUTPUT_VIDEO_DIR, OUTPUT_H5_DIR, ensure_output_dirs

ensure_output_dirs()

url_1 = "rtsp://ubnt:ubnt@192.168.5.41:554/s2"
url_2 = "rtsp://ubnt:ubnt@192.168.5.45:554/s2"
url_3 = "rtsp://ubnt:ubnt@192.168.5.42:554/s2"
url_4 = "rtsp://ubnt:ubnt@192.168.5.48:554/s2"



cap_1 = cv2.VideoCapture(url_1, cv2.CAP_FFMPEG)
cap_2 = cv2.VideoCapture(url_2, cv2.CAP_FFMPEG)
cap_3 = cv2.VideoCapture(url_3, cv2.CAP_FFMPEG)
cap_4 = cv2.VideoCapture(url_4, cv2.CAP_FFMPEG)
cv2.setUseOptimized(True)

# Detection and tracking models
RTMDET_MODEL = 'rtmdet-m-640.onnx'
RTMPOSE_MODEL = 'rtmpose-m-256-192.onnx'
# RTMPose engine
device = 'cuda'
backend = 'onnxruntime'

RTMDET_MODEL = os.path.join(MODEL_DIR, RTMDET_MODEL)
RTMPOSE_MODEL = os.path.join(MODEL_DIR, RTMPOSE_MODEL)

detector = RTMDet(
    onnx_model=RTMDET_MODEL,
    model_input_size=(640, 640),
    backend=backend,
    device=device
)

pose_estimator = RTMPose(
            onnx_model=RTMPOSE_MODEL,
            model_input_size = (192, 256),            
            backend=backend,
            device=device)

def stitch_frames(frame_1, frame_2, frame_3, frame_4):
    # Resize each frame to 960x540
    size = (960, 540)
    f1 = cv2.resize(frame_1, size)
    f2 = cv2.resize(frame_2, size)
    f3 = cv2.resize(frame_3, size)
    f4 = cv2.resize(frame_4, size)

    # Top row: f1 | f2
    top = np.hstack((f1, f2))
    # Bottom row: f3 | f4
    bottom = np.hstack((f3, f4))

    # Combine rows
    combined = np.vstack((top, bottom))

    return combined

def run_pose(frame, detector, pose_estimator):
    det_bboxes, det_scores = detector(frame)
    keypoints_list, scores_list = pose_estimator(frame, det_bboxes)
    frame = draw_skeleton(
            frame,
            np.array(keypoints_list),        # shape (1, K, 2)
            np.array(scores_list),       # shape (1, K)
            openpose_skeleton=False,
            kpt_thr=0.3,
            radius=3,
            line_width=2
        )
    return frame

while True:
    start_time = time.time()
    ret, frame_1 = cap_1.read()
    ret, frame_2 = cap_2.read()
    ret, frame_3 = cap_3.read()
    ret, frame_4 = cap_4.read()
    cap_time = time.time()

    
    # frame_1 = run_pose(frame_1, detector, pose_estimator)
    # frame_2 = run_pose(frame_2, detector, pose_estimator)
    # frame_3 = run_pose(frame_3, detector, pose_estimator)
    # frame_4 = run_pose(frame_4, detector, pose_estimator)
    det_time = time.time() 

    #combine all 4 streams into single 
    stitched_img = stitch_frames(frame_1, frame_2, frame_3, frame_4)

    # Timing info
    cap_duration = (cap_time - start_time) * 1000
    det_duration = (det_time - cap_time) * 1000

    frame = cv2.putText(stitched_img, f'cap: {cap_duration: .2f} ms', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    frame = cv2.putText(stitched_img, f'det: {det_duration: .2f} ms', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
    cv2.imshow("Camera", stitched_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_1.release()
cap_2.release()
cap_3.release()
cap_4.release()
cv2.destroyAllWindows()