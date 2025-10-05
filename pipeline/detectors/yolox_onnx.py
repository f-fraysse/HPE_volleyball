"""
YOLOX ONNX detector adapter for HPE_volleyball pipeline.

This implements YOLOX detection using ONNX Runtime directly.
Based on Megvii-BaseDetection/YOLOX preprocessing and postprocessing.
"""

import time
import numpy as np
import onnxruntime as ort
import cv2


class YOLOXONNXDetector:
    """
    YOLOX detector using ONNX Runtime.

    Handles preprocessing, inference, and postprocessing for YOLOX models.
    """

    def __init__(self, model_path: str, device: str = "cuda", backend: str = "onnxruntime",
                 model_input_size: tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.7, ball_conf_threshold: float = 0.3,
                 nms_iou_threshold: float = 0.45):
        """
        Initialize YOLOX detector.

        Args:
            model_path: Path to YOLOX ONNX model
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

        # Debug flag for first frame
        self.debug_first_frame = True

    def __call__(self, frame: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
        """
        Process a single frame with YOLOX.

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

        # Inference (prep is minimal for ONNX - just dict creation)
        prep_start = time.perf_counter()
        input_feed = {self.input_name: input_tensor}
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
        Preprocess frame for YOLOX (letterbox resize, no normalization).

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

        # Create padded image (model_h, model_w, 3) with gray padding (114)
        padded = np.full((self.model_h, self.model_w, 3), 114, dtype=np.uint8)
        pad_w = (self.model_w - new_w) // 2
        pad_h = (self.model_h - new_h) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # Convert to float32 (YOLOX expects 0-255 range, no normalization)
        padded = padded.astype(np.float32, copy=False)

        # HWC to CHW
        input_tensor = np.transpose(padded, (2, 0, 1))

        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, 0)

        # Scale and pad info for postprocessing
        scale = r
        pad = (pad_w, pad_h, new_w, new_h)

        return input_tensor, scale, pad

    def _postprocess(self, outputs: list[np.ndarray], scale: float, pad: tuple[int, int, int, int],
                    original_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess YOLOX outputs with stride-based grid decoding.

        Args:
            outputs: Model outputs [batch, num_predictions, 85]
                     85 = [grid_x, grid_y, log_w, log_h, objectness, 80_class_scores]
            scale: Scale factor from preprocessing
            pad: Padding info (pad_w, pad_h, new_w, new_h)
            original_shape: Original frame shape (H, W)

        Returns:
            Tuple of (bboxes_xyxy, scores, class_labels) for person and sports ball classes
        """
        # YOLOX output shape: [1, num_predictions, 85]
        predictions = outputs[0]  # Shape: [1, num_predictions, 85]
        
        # Check if we need to squeeze batch dimension
        if predictions.ndim == 3:
            predictions = predictions[0]  # Remove batch dimension: [num_predictions, 85]

        # Apply stride-based grid decoding (from YOLOX demo_postprocess)
        predictions = self._decode_outputs(predictions, (self.model_h, self.model_w))

        # Extract components after decoding
        boxes_cxcywh = predictions[:, :4]  # [cx, cy, w, h] now in letterbox pixel space
        objectness = predictions[:, 4]      # objectness score
        class_scores = predictions[:, 5:]   # 80 class scores

        # Calculate final scores: objectness * class_score
        person_scores = objectness * class_scores[:, 0]
        ball_scores = objectness * class_scores[:, 32]  # Sports ball is class 32

        # Convert center format to xyxy in letterbox space
        boxes_xyxy = np.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

        # First-frame debug logging
        if self.debug_first_frame:
            print(f"[YOLOX] Original frame: H={original_shape[0]}, W={original_shape[1]}")
            print(f"[YOLOX] Model input: H={self.model_h}, W={self.model_w}")
            print(f"[YOLOX] Preprocessing: scale={scale:.3f}, pad=(w:{pad[0]}, h:{pad[1]}, new_w:{pad[2]}, new_h:{pad[3]})")
            print(f"[YOLOX] Raw predictions: {predictions.shape}")
            print(f"[YOLOX] Objectness range: [{objectness.min():.4f}, {objectness.max():.4f}]")
            print(f"[YOLOX] Class scores shape: {class_scores.shape}")
            print(f"[YOLOX] Person scores range: [{person_scores.min():.4f}, {person_scores.max():.4f}]")
            print(f"[YOLOX] Raw cxcywh range: cx=[{boxes_cxcywh[:,0].min():.1f}, {boxes_cxcywh[:,0].max():.1f}], "
                  f"cy=[{boxes_cxcywh[:,1].min():.1f}, {boxes_cxcywh[:,1].max():.1f}], "
                  f"w=[{boxes_cxcywh[:,2].min():.1f}, {boxes_cxcywh[:,2].max():.1f}], "
                  f"h=[{boxes_cxcywh[:,3].min():.1f}, {boxes_cxcywh[:,3].max():.1f}]")
            print(f"[YOLOX] Raw boxes range: x1=[{boxes_xyxy[:,0].min():.1f}, {boxes_xyxy[:,0].max():.1f}], "
                  f"y1=[{boxes_xyxy[:,1].min():.1f}, {boxes_xyxy[:,1].max():.1f}], "
                  f"x2=[{boxes_xyxy[:,2].min():.1f}, {boxes_xyxy[:,2].max():.1f}], "
                  f"y2=[{boxes_xyxy[:,3].min():.1f}, {boxes_xyxy[:,3].max():.1f}]")

        # Reverse letterbox transformation
        pad_w, pad_h, new_w, new_h = pad
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale

        if self.debug_first_frame:
            print(f"[YOLOX] After reverse-letterbox: x1=[{boxes_xyxy[:,0].min():.1f}, {boxes_xyxy[:,0].max():.1f}], "
                  f"y1=[{boxes_xyxy[:,1].min():.1f}, {boxes_xyxy[:,1].max():.1f}], "
                  f"x2=[{boxes_xyxy[:,2].min():.1f}, {boxes_xyxy[:,2].max():.1f}], "
                  f"y2=[{boxes_xyxy[:,3].min():.1f}, {boxes_xyxy[:,3].max():.1f}]")

        # Clip to original frame bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, original_shape[1])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, original_shape[0])

        # Create arrays for person and ball detections
        all_boxes = []
        all_scores = []
        all_labels = []

        # Filter person detections
        person_mask = person_scores >= self.conf_threshold
        if person_mask.any():
            all_boxes.append(boxes_xyxy[person_mask])
            all_scores.append(person_scores[person_mask])
            all_labels.append(np.zeros(person_mask.sum(), dtype=np.int32))

        # Filter ball detections
        ball_mask = ball_scores >= self.ball_conf_threshold
        if ball_mask.any():
            all_boxes.append(boxes_xyxy[ball_mask])
            all_scores.append(ball_scores[ball_mask])
            all_labels.append(np.full(ball_mask.sum(), 32, dtype=np.int32))

        # Combine all detections
        if len(all_boxes) == 0:
            if self.debug_first_frame:
                print("[YOLOX] No boxes after confidence filtering")
                self.debug_first_frame = False
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        combined_boxes = np.vstack(all_boxes)
        combined_scores = np.concatenate(all_scores)
        combined_labels = np.concatenate(all_labels)

        # Apply NMS separately per class to avoid suppressing balls near persons
        final_bboxes = []
        final_scores = []
        final_labels = []

        for class_id in [0, 32]:  # Person and sports ball
            class_mask = combined_labels == class_id
            if not class_mask.any():
                continue

            class_bboxes = combined_boxes[class_mask]
            class_scores = combined_scores[class_mask]

            indices = self._nms(class_bboxes, class_scores, self.nms_iou_threshold)
            final_bboxes.append(class_bboxes[indices])
            final_scores.append(class_scores[indices])
            final_labels.append(np.full(len(indices), class_id, dtype=np.int32))

        if len(final_bboxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        bboxes = np.vstack(final_bboxes)
        scores = np.concatenate(final_scores)
        labels = np.concatenate(final_labels)

        # Final assertions for first frame
        if self.debug_first_frame:
            # Check for valid boxes
            valid_boxes = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1]) & \
                         (bboxes[:, 0] >= 0) & (bboxes[:, 1] >= 0) & \
                         (bboxes[:, 2] <= original_shape[1]) & (bboxes[:, 3] <= original_shape[0])
            if not np.all(valid_boxes):
                invalid_count = np.sum(~valid_boxes)
                print(f"[YOLOX] WARNING: {invalid_count} invalid boxes after postprocessing")
            else:
                print(f"[YOLOX] All {len(bboxes)} boxes are valid after postprocessing")

            person_count = np.sum(labels == 0)
            ball_count = np.sum(labels == 32)
            print(f"[YOLOX] Detected {person_count} persons and {ball_count} sports balls")
            self.debug_first_frame = False

        return bboxes, scores, labels

    def _decode_outputs(self, outputs: np.ndarray, img_size: tuple[int, int], p6: bool = False) -> np.ndarray:
        """
        Decode YOLOX outputs using stride-based grids.
        
        Based on YOLOX demo_postprocess function.
        
        Args:
            outputs: Raw model outputs [num_predictions, 85]
            img_size: Model input size (height, width)
            p6: Whether model uses P6 in FPN/PAN (default False for standard models)
            
        Returns:
            Decoded predictions with absolute coordinates in letterbox space
        """
        grids = []
        expanded_strides = []

        # Standard YOLOX uses strides [8, 16, 32]
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        
        # Expand dimensions to match outputs
        grids = grids[0]  # Remove batch dimension
        expanded_strides = expanded_strides[0]  # Remove batch dimension
        
        # Decode box coordinates
        # outputs[..., :2] are grid-relative offsets, add grid position and multiply by stride
        outputs[:, :2] = (outputs[:, :2] + grids) * expanded_strides
        # outputs[..., 2:4] are log-space width/height, exp and multiply by stride
        outputs[:, 2:4] = np.exp(outputs[:, 2:4]) * expanded_strides

        return outputs

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
