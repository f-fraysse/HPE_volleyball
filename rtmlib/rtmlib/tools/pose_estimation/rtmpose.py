from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose, get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine


class RTMPose(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (288, 384),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        # Store model input size as numpy array for vectorized postprocessing
        self.model_input_size_np = np.array(self.model_input_size)

    def __call__(self, image: np.ndarray, bboxes: list = []):
        import time
        total_start = time.perf_counter()

        # Handle case with no bounding boxes
        if not bboxes:
            num_keypoints = 17 # Default to COCO 17 keypoints if model info isn't readily available
            # Attempt to get num_keypoints from model output shape if possible (might need adjustment)
            try:
                # This assumes the model output shape is known or can be inferred
                # Example: if outputs[0] shape is (N, K, W_simcc), K is num_keypoints
                # This part might need refinement based on how model info is stored/accessed
                pass # Placeholder: Add logic to get K if needed
            except Exception:
                pass
            timing_info = {
                'total': (time.perf_counter() - total_start) * 1000,
                'preprocess': 0, 'prep': 0, 'model': 0, 'postprocess': 0, 'num_bboxes': 0
            }
            return np.empty((0, num_keypoints, 2)), np.empty((0, num_keypoints)), timing_info

        # --- Batch Preprocessing ---
        preprocess_start = time.perf_counter()
        batch_img_np, centers, scales = self.preprocess(image, bboxes)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000

        # --- Batch Inference ---
        inference_start = time.perf_counter()
        # Transpose the batch from NHWC (output of preprocess) to NCHW for ONNX Runtime
        batch_input = batch_img_np.transpose(0, 3, 1, 2)
        # Ensure contiguous array for ONNX Runtime
        batch_input = np.ascontiguousarray(batch_input, dtype=np.float32)
        # Call inference ONCE with the batch
        outputs = self.inference(batch_input) # BaseTool.inference handles the dict creation
        inference_time = (time.perf_counter() - inference_start) * 1000
        # Extract detailed timing from the single inference call
        # Assuming _last_inference_timing is updated correctly by base.inference
        prep_timing = self._last_inference_timing.get('prep', 0)
        model_timing = self._last_inference_timing.get('model', 0)

        # --- Batch Postprocessing ---
        postprocess_start = time.perf_counter()
        keypoints, scores = self.postprocess(outputs, centers, scales) # Pass lists of centers/scales
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000

        # --- Final Steps ---
        if self.to_openpose:
             # Assuming convert_coco_to_openpose handles batch input/output
             # If not, a loop might be needed here as well
             keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        total_time = (time.perf_counter() - total_start) * 1000

        timing_info = {
            'total': total_time,
            'preprocess': preprocess_time, # Total time for batch preprocess
            'prep': prep_timing,           # Prep time from the single inference call
            'model': model_timing,         # Model time from the single inference call
            'postprocess': postprocess_time, # Total time for batch postprocess
            'num_bboxes': len(bboxes)
        }

        return keypoints, scores, timing_info

    def preprocess(self, img: np.ndarray, bboxes: list):
        """Do preprocessing for RTMPose model inference for a batch of bounding boxes.

        Args:
            img (np.ndarray): Input image in shape (H, W, C).
            bboxes (list): A list of xyxy-format bounding boxes.

        Returns:
            tuple:
            - batch_img_np (np.ndarray): Batch of preprocessed images (N, H, W, C).
            - centers (list): List of centers corresponding to each bbox.
            - scales (list): List of scales corresponding to each bbox.
        """
        batch_img = []
        centers = []
        scales = []

        for bbox in bboxes:
            bbox_np = np.array(bbox)
            # get center and scale
            center, scale = bbox_xyxy2cs(bbox_np, padding=1.25)

            # do affine transformation
            # top_down_affine returns the transformed image crop
            resized_img, updated_scale = top_down_affine(self.model_input_size, scale, center, img)

            # normalize image
            if self.mean is not None:
                # Use float32 for normalization
                resized_img_float = resized_img.astype(np.float32, copy=True) # Use copy=True to avoid modifying previous loop items if needed
                cv2.subtract(resized_img_float, self.mean, dst=resized_img_float)
                cv2.divide(resized_img_float, self.std, dst=resized_img_float)
                batch_img.append(resized_img_float)
            else:
                 batch_img.append(resized_img) # Append original uint8 if no normalization

            centers.append(center)
            scales.append(updated_scale) # Store the scale used for the transformation

        # Stack the processed images into a batch
        if batch_img:
            batch_img_np = np.stack(batch_img, axis=0)
        else:
            # Handle case with no bounding boxes
            batch_img_np = np.empty((0, self.model_input_size[0], self.model_input_size[1], 3), dtype=np.float32)

        return batch_img_np, centers, scales

    def postprocess(
            self,
            outputs: List[np.ndarray], # Expects BATCHED outputs [batch_simcc_x, batch_simcc_y]
            centers: list,             # List of centers
            scales: list,              # List of scales
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output for a batch.

        Args:
            outputs (List[np.ndarray]): Batched output of RTMPose model.
                                        [0]: simcc_x (N, K, Ws)
                                        [1]: simcc_y (N, K, Hs)
            centers (list): List of centers corresponding to the batch.
            scales (list): List of scales corresponding to the batch.
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - final_keypoints (np.ndarray): Batch of rescaled keypoints (N, K, 2).
            - final_scores (np.ndarray): Batch of model predict scores (N, K).
        """
        batch_simcc_x, batch_simcc_y = outputs
        num_bboxes = batch_simcc_x.shape[0]
        num_keypoints = batch_simcc_x.shape[1] # K

        if num_bboxes == 0:
            return np.empty((0, num_keypoints, 2)), np.empty((0, num_keypoints))

        # Process the batch using loops (assuming get_simcc_maximum is not batched)
        batch_keypoints_list = []
        batch_scores_list = []

        # Pre-convert lists to numpy arrays for efficiency inside loop if needed
        centers_np = np.array(centers) # Shape (N, 2)
        scales_np = np.array(scales)   # Shape (N, 2)

        for i in range(num_bboxes):
            # Process each item in the batch
            simcc_x = batch_simcc_x[i:i+1] # Keep batch dim for get_simcc_maximum (N=1, K, Ws)
            simcc_y = batch_simcc_y[i:i+1] # Keep batch dim for get_simcc_maximum (N=1, K, Hs)
            center = centers_np[i] # Shape (2,)
            scale = scales_np[i]   # Shape (2,)

            # Decode simcc for one item
            # get_simcc_maximum expects batch input (N, K, Ws/Hs)
            locs, scores = get_simcc_maximum(simcc_x, simcc_y)
            # locs shape: (1, K, 2), scores shape: (1, K)
            locs = locs[0] # Remove batch dim -> (K, 2)
            scores = scores[0] # Remove batch dim -> (K,)

            keypoints = locs / simcc_split_ratio

            # Rescale keypoints for one item
            keypoints = keypoints / self.model_input_size_np * scale
            keypoints = keypoints + center - scale / 2

            batch_keypoints_list.append(keypoints)
            batch_scores_list.append(scores)

        # Stack results at the end
        final_keypoints = np.stack(batch_keypoints_list, axis=0) # Shape (N, K, 2)
        final_scores = np.stack(batch_scores_list, axis=0)       # Shape (N, K)

        return final_keypoints, final_scores
