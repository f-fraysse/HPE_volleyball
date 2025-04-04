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

    def __call__(self, image: np.ndarray, bboxes: list = []):
        # Add timing for the entire pose estimation process
        import time
        total_start = time.perf_counter()
        
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores = [], []
        
        # Track timing for each bbox separately
        preprocess_times = []
        inference_times = []
        postprocess_times = []
        
        for bbox in bboxes:
            # Preprocessing timing
            preprocess_start = time.perf_counter()
            img, center, scale = self.preprocess(image, bbox)
            preprocess_time = (time.perf_counter() - preprocess_start) * 1000
            preprocess_times.append(preprocess_time)
            
            # Inference timing
            inference_start = time.perf_counter()
            outputs = self.inference(img)
            inference_time = (time.perf_counter() - inference_start) * 1000
            inference_times.append(inference_time)
            
            # Postprocessing timing
            postprocess_start = time.perf_counter()
            kpts, score = self.postprocess(outputs, center, scale)
            postprocess_time = (time.perf_counter() - postprocess_start) * 1000
            postprocess_times.append(postprocess_time)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Calculate average times if multiple bboxes were processed
        avg_preprocess = sum(preprocess_times) / len(preprocess_times) if preprocess_times else 0
        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_postprocess = sum(postprocess_times) / len(postprocess_times) if postprocess_times else 0
        
        # Store timing information in a dictionary
        timing_info = {
            'total': total_time,
            'preprocess': avg_preprocess,
            'prep': self._last_inference_timing['prep'],
            'model': self._last_inference_timing['model'],
            'postprocess': avg_postprocess,
            'num_bboxes': len(bboxes)
        }
        
        # Return results along with timing information
        return keypoints, scores, timing_info

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)
        # normalize image
        if self.mean is not None:
            #normalise with openCV - 2x faster
            resized_img = resized_img.astype(np.float32, copy=False)
            cv2.subtract(resized_img, self.mean, dst=resized_img)
            cv2.divide(resized_img, self.std, dst=resized_img)

        return resized_img, center, scale

    def postprocess(
            self,
            outputs: List[np.ndarray],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores
