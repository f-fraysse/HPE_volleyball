"""
Base interface for detection components in the HPE_volleyball pipeline.

All detectors must implement the Detector protocol to ensure consistent
interface for the main pipeline loop.
"""

from typing import Protocol, Tuple, Dict
import numpy as np


class Detector(Protocol):
    """
    Protocol for detection components.

    Detectors process a single frame and return bounding boxes, scores, and timing info.
    """

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Process a single frame and return detections.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format (OpenCV convention)

        Returns:
            Tuple of:
            - bboxes_xyxy: np.ndarray of shape (N, 4) with [x1, y1, x2, y2] in absolute image coordinates
            - scores: np.ndarray of shape (N,) with confidence scores
            - timing: Dict with timing info, must include keys:
                - 'total': total detection time
                - 'preprocess': preprocessing time
                - 'prep': data preparation/transfer time
                - 'model': model inference time
                - 'postprocess': postprocessing time
        """
        ...


def create_detector(detector_type: str, model_path: str, device: str = "cuda", backend: str = "onnxruntime",
                   ball_conf_threshold: float = 0.5) -> Detector:
    """
    Factory function to create detector instances.

    Args:
        detector_type: Type of detector ("rtmdet", "rtdetr", or "yolox")
        model_path: Path to the ONNX model file
        device: Device for inference ("cuda" or "cpu")
        backend: Backend for inference ("onnxruntime")
        ball_conf_threshold: Confidence threshold for sports ball detections (rtdetr and yolox)

    Returns:
        Detector instance

    Raises:
        ValueError: If detector_type is not supported
    """
    if detector_type == "rtmdet":
        from .detectors.rtmdet_onnx import RTMDetONNXDetector
        return RTMDetONNXDetector(model_path, device, backend)
    elif detector_type == "rtdetr":
        from .detectors.rtdetr_onnx import RTDetrONNXDetector
        return RTDetrONNXDetector(model_path, device, backend, ball_conf_threshold=ball_conf_threshold)
    elif detector_type == "yolox":
        from .detectors.yolox_onnx import YOLOXONNXDetector
        return YOLOXONNXDetector(model_path, device, backend, ball_conf_threshold=ball_conf_threshold)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}. Supported: 'rtmdet', 'rtdetr', 'yolox'")
