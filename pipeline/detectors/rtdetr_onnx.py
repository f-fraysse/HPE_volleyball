"""
RT-DETR ONNX detector adapter for HPE_volleyball pipeline.

This implements RT-DETR detection using ONNX Runtime directly.
Based on lyuwenyu/RT-DETR preprocessing and postprocessing.
"""

import time
import numpy as np
import onnxruntime as ort
import cv2


class RTDetrONNXDetector:
    """
    RT-DETR detector using ONNX Runtime.

    Handles preprocessing, inference, and postprocessing for RT-DETR models.
    """

    def __init__(self, model_path: str, device: str = "cuda", backend: str = "onnxruntime",
                 model_input_size: tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.7, ball_conf_threshold: float = 0.3,
                 nms_iou_threshold: float = 0.45):
        """
        Initialize RT-DETR detector.

        Args:
            model_path: Path to RT-DETR ONNX model
            device: Device for inference ("cuda" or "cpu")
            backend: Backend ("onnxruntime")
            model_input_size: Input size for the model (width, height)
            conf_threshold: Confidence threshold for person detections
            ball_conf_threshold: Confidence threshold for sports ball detections
            nms_iou_threshold: IoU threshold for NMS
        """
        self.model_w, self.model_h = model_input_size
        self.conf_threshold = conf_threshold
        self.ball_conf_threshold = ball_conf_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # Set up ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # RT-DETR normalization constants optimized for cv2 operations
        # Converted from RGB [0.485, 0.456, 0.406] * 255 = [123.675, 116.28, 103.53]
        # Reversed to BGR format and scaled to 0-255 range for cv2 operations
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32)

        # Debug flag for first frame
        self.debug_first_frame = True

    def __call__(self, frame: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
        """
        Process a single frame with RT-DETR.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format

        Returns:
            Tuple of ((bboxes_xyxy, scores, class_labels), timing_dict)
        """
        start_time = time.perf_counter()

        # Preprocessing
        preprocess_start = time.perf_counter()
        input_tensor, scale, pad = self._preprocess(frame)
        preprocess_time = time.perf_counter() - preprocess_start

        # Inference (prep is building input feed dict)
        prep_start = time.perf_counter()

        # Build input feed for ORT
        input_feed = {self.input_name: input_tensor}

        # Add orig_target_sizes - try using model input size instead of original frame size
        model_input_names = [inp.name for inp in self.session.get_inputs()]
        if "orig_target_sizes" in model_input_names:
            # Try model input size instead of original frame size
            model_sizes = np.array([[self.model_h, self.model_w]], dtype=np.int64)
            input_feed["orig_target_sizes"] = model_sizes

        prep_time = time.perf_counter() - prep_start

        # Model inference
        model_start = time.perf_counter()
        outputs = self.session.run(self.output_names, input_feed)
        model_time = time.perf_counter() - model_start

        # Postprocessing
        postprocess_start = time.perf_counter()
        bboxes_xyxy, scores, class_labels = self._postprocess(outputs, scale, pad, frame.shape[:2])
        postprocess_time = time.perf_counter() - postprocess_start

        total_time = time.perf_counter() - start_time

        # Convert all timings to milliseconds
        timing = {
            'total': total_time * 1000,
            'preprocess': preprocess_time * 1000,
            'prep': prep_time * 1000,
            'model': model_time * 1000,
            'postprocess': postprocess_time * 1000
        }

        return (bboxes_xyxy, scores, class_labels), timing

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
        """
        Preprocess frame for RT-DETR.

        Args:
            frame: Input frame (H, W, C) BGR

        Returns:
            Tuple of (input_tensor, scale, pad_info)
        """
        # Get original dimensions
        h, w = frame.shape[:2]

        # Letterbox resize to model input size (model_w, model_h)
        r = min(self.model_w / w, self.model_h / h)
        new_w, new_h = int(w * r), int(h * r)

        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image (model_h, model_w, 3)
        padded = np.full((self.model_h, self.model_w, 3), 114, dtype=np.uint8)
        pad_w = (self.model_w - new_w) // 2
        pad_h = (self.model_h - new_h) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # Convert to float32 without copy, no division by 255 (stays in 0-255 range)
        # Using copy=False for in-place conversion to avoid memory allocation
        padded = padded.astype(np.float32, copy=False)

        # Normalize using cv2 in-place operations (3x faster than NumPy)
        # Stays in BGR format - no color conversion needed
        cv2.subtract(padded, self.mean, dst=padded)
        cv2.divide(padded, self.std, dst=padded)

        # HWC to CHW
        input_tensor = np.transpose(padded, (2, 0, 1))

        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, 0)

        # Scale and pad info for postprocessing
        scale = r
        pad = (pad_w, pad_h, new_w, new_h)

        return input_tensor.astype(np.float32), scale, pad

    def _postprocess(self, outputs: list[np.ndarray], scale: float, pad: tuple[int, int, int, int],
                    original_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess RT-DETR outputs with robust coordinate space handling.

        Args:
            outputs: Model outputs
            scale: Scale factor from preprocessing
            pad: Padding info (pad_w, pad_h, new_w, new_h)
            original_shape: Original frame shape (H, W)

        Returns:
            Tuple of (bboxes_xyxy, scores, class_labels) for person and sports ball classes
        """
        # RT-DETR outputs: ['labels', 'boxes', 'scores']
        labels = outputs[0][0]  # Shape: (num_queries,) - class labels
        boxes = outputs[1][0]   # Shape: (num_queries, 4) - xyxy coordinates
        scores = outputs[2][0]  # Shape: (num_queries,) - confidence scores

        # Get person (class 0) and sports ball (class 32)
        target_mask = (labels == 0) | (labels == 32)
        if not target_mask.any():
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        target_boxes = boxes[target_mask]
        target_scores = scores[target_mask]
        target_labels = labels[target_mask]

        # First-frame debug logging
        if self.debug_first_frame:
            print(f"[RT-DETR] Original frame: H={original_shape[0]}, W={original_shape[1]}")
            print(f"[RT-DETR] Model input: H={self.model_h}, W={self.model_w}")
            print(f"[RT-DETR] Preprocessing: scale={scale:.3f}, pad=(w:{pad[0]}, h:{pad[1]}, new_w:{pad[2]}, new_h:{pad[3]})")
            print(f"[RT-DETR] Raw boxes range: x1=[{target_boxes[:,0].min():.1f}, {target_boxes[:,0].max():.1f}], "
                  f"y1=[{target_boxes[:,1].min():.1f}, {target_boxes[:,1].max():.1f}], "
                  f"x2=[{target_boxes[:,2].min():.1f}, {target_boxes[:,2].max():.1f}], "
                  f"y2=[{target_boxes[:,3].min():.1f}, {target_boxes[:,3].max():.1f}]")

        # RT-DETR models typically output letterbox-space coordinates
        # Always apply reverse-letterbox transformation
        pad_w, pad_h, new_w, new_h = pad
        target_boxes[:, [0, 2]] = (target_boxes[:, [0, 2]] - pad_w) / scale
        target_boxes[:, [1, 3]] = (target_boxes[:, [1, 3]] - pad_h) / scale

        if self.debug_first_frame:
            print(f"[RT-DETR] After reverse-letterbox: x1=[{target_boxes[:,0].min():.1f}, {target_boxes[:,0].max():.1f}], "
                  f"y1=[{target_boxes[:,1].min():.1f}, {target_boxes[:,1].max():.1f}], "
                  f"x2=[{target_boxes[:,2].min():.1f}, {target_boxes[:,2].max():.1f}], "
                  f"y2=[{target_boxes[:,3].min():.1f}, {target_boxes[:,3].max():.1f}]")

        # Clip to original frame bounds
        bboxes = target_boxes.copy()
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, original_shape[1])
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, original_shape[0])

        # Filter by confidence (different thresholds for person vs ball)
        person_mask = target_labels == 0
        ball_mask = target_labels == 32
        
        person_conf_mask = person_mask & (target_scores >= self.conf_threshold)
        ball_conf_mask = ball_mask & (target_scores >= self.ball_conf_threshold)
        conf_mask = person_conf_mask | ball_conf_mask
        
        bboxes = bboxes[conf_mask]
        target_scores = target_scores[conf_mask]
        target_labels = target_labels[conf_mask]

        if len(bboxes) == 0:
            if self.debug_first_frame:
                print("[RT-DETR] No boxes after confidence filtering")
                self.debug_first_frame = False
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        # Apply NMS separately per class to avoid suppressing balls near persons
        final_bboxes = []
        final_scores = []
        final_labels = []
        
        for class_id in [0, 32]:  # Person and sports ball
            class_mask = target_labels == class_id
            if not class_mask.any():
                continue
                
            class_bboxes = bboxes[class_mask]
            class_scores = target_scores[class_mask]
            
            indices = self._nms(class_bboxes, class_scores, self.nms_iou_threshold)
            final_bboxes.append(class_bboxes[indices])
            final_scores.append(class_scores[indices])
            final_labels.append(np.full(len(indices), class_id, dtype=np.int32))
        
        if len(final_bboxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
            
        bboxes = np.vstack(final_bboxes)
        target_scores = np.concatenate(final_scores)
        target_labels = np.concatenate(final_labels)

        # Final assertions for first frame
        if self.debug_first_frame:
            # Check for valid boxes
            valid_boxes = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1]) & \
                         (bboxes[:, 0] >= 0) & (bboxes[:, 1] >= 0) & \
                         (bboxes[:, 2] <= original_shape[1]) & (bboxes[:, 3] <= original_shape[0])
            if not np.all(valid_boxes):
                invalid_count = np.sum(~valid_boxes)
                print(f"[RT-DETR] WARNING: {invalid_count} invalid boxes after postprocessing")
            else:
                print(f"[RT-DETR] All {len(bboxes)} boxes are valid after postprocessing")
            
            person_count = np.sum(target_labels == 0)
            ball_count = np.sum(target_labels == 32)
            print(f"[RT-DETR] Detected {person_count} persons and {ball_count} sports balls")
            self.debug_first_frame = False

        return bboxes, target_scores, target_labels

    def _detect_coordinate_space(self, boxes: np.ndarray, original_shape: tuple[int, int]) -> str:
        """
        Auto-detect the coordinate space of bounding boxes.

        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            original_shape: Original frame shape (H, W)

        Returns:
            'normalized', 'model_letterbox', or 'absolute'
        """
        if boxes.size == 0:
            return 'absolute'

        max_coord = boxes.max()
        orig_h, orig_w = original_shape

        # Check for normalized coordinates (0..1)
        if max_coord <= 1.5:
            return 'normalized'

        # Check if coordinates are within model input bounds (likely letterbox space)
        # Allow some tolerance for floating point precision
        model_max = max(self.model_w, self.model_h) + 50  # +50 for padding tolerance
        if max_coord <= model_max:
            # Additional check: are there boxes near the padding edges?
            # This is a heuristic - if many boxes are close to model edges, likely letterbox space
            edge_threshold = 20
            near_edges = ((boxes[:, 0] < edge_threshold) | (boxes[:, 1] < edge_threshold) |
                         (boxes[:, 2] > self.model_w - edge_threshold) |
                         (boxes[:, 3] > self.model_h - edge_threshold))
            if np.mean(near_edges) > 0.3:  # More than 30% of boxes near edges
                return 'model_letterbox'

        # Otherwise, assume absolute coordinates
        return 'absolute'

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Non-maximum suppression.

        Args:
            boxes: Bounding boxes (N, 4)
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold

        Returns:
            Indices of kept boxes
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)
