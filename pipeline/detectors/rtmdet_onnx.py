"""
RTMDet ONNX detector adapter for HPE_volleyball pipeline.

This is a thin wrapper around rtmlib.RTMDet to match the Detector protocol.
"""

import time
import numpy as np
from rtmlib import RTMDet


class RTMDetONNXDetector:
    """
    RTMDet detector using ONNX Runtime via rtmlib.

    This preserves the current detection behavior while conforming to the Detector protocol.
    """

    def __init__(self, model_path: str, device: str = "cuda", backend: str = "onnxruntime",
                 model_input_size: tuple[int, int] = (640, 640)):
        """
        Initialize RTMDet detector.

        Args:
            model_path: Path to RTMDet ONNX model
            device: Device for inference ("cuda" or "cpu")
            backend: Backend ("onnxruntime")
            model_input_size: Input size for the model (width, height)
        """
        self.detector = RTMDet(
            onnx_model=model_path,
            model_input_size=model_input_size,
            backend=backend,
            device=device
        )

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Process a single frame with RTMDet.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format

        Returns:
            Tuple of (bboxes_xyxy, scores, timing_dict)
        """
        start_time = time.perf_counter()

        # Call RTMDet detector
        bboxes_scores, timing = self.detector(frame)

        # RTMDet returns (bboxes, scores) where bboxes are already xyxy
        bboxes_xyxy, scores = bboxes_scores

        total_time = time.perf_counter() - start_time

        # Ensure timing dict has all required keys (rtmlib may provide some)
        full_timing = {
            'total': total_time,
            'preprocess': timing.get('preprocess', 0.0),
            'prep': timing.get('prep', 0.0),
            'model': timing.get('model', 0.0),
            'postprocess': timing.get('postprocess', 0.0)
        }

        return (bboxes_xyxy, scores), full_timing
