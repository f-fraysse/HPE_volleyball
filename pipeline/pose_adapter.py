"""
Pose estimator adapter for HPE_volleyball pipeline.

Wrapper around rtmlib.RTMPose to provide a clean interface.
"""

import numpy as np
from rtmlib import RTMPose


def create_pose_estimator(model_path: str, device: str = "cuda", backend: str = "onnxruntime",
                         model_input_size: tuple[int, int] = (192, 256)) -> RTMPose:
    """
    Create RTMPose instance.

    Args:
        model_path: Path to RTMPose ONNX model
        device: Device for inference
        backend: Backend
        model_input_size: Model input size (height, width)

    Returns:
        RTMPose instance
    """
    return RTMPose(
        onnx_model=model_path,
        model_input_size=model_input_size,
        backend=backend,
        device=device
    )


def estimate_pose(pose_estimator: RTMPose, frame: np.ndarray, bboxes: list[list[float]]) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    """
    Estimate poses for given bounding boxes.

    Args:
        pose_estimator: RTMPose instance
        frame: Input frame
        bboxes: List of bounding boxes [x1, y1, x2, y2]

    Returns:
        Tuple of (keypoints_list, scores_list, timing_dict)
    """
    if not bboxes:  # Check if empty list
        return [], [], {'total': 0.0, 'preprocess': 0.0, 'prep': 0.0, 'model': 0.0, 'postprocess': 0.0}

    # RTMPose expects bboxes as list of lists (not numpy array)
    # Call RTMPose with original format
    keypoints_list, scores_list, timing = pose_estimator(frame, bboxes)

    return keypoints_list, scores_list, timing
