# Active Context: HPE_volleyball

## Current Work Focus

The current focus of the HPE_volleyball project is **performance optimization** of the inference pipeline. While the detection-tracking-pose pipeline is functionally complete and produces accurate results, the processing speed needs significant improvement to make the system more practical for real-world use.

We have implemented a **modular detector architecture** to enable easy swapping between detection models while maintaining consistent interfaces and profiling. The pipeline now supports RTMDet (baseline) and RT-DETR (alternative) detectors running on ONNX Runtime.

In addition to optimizing the existing RTMDet/RTMPose pipeline, we are also exploring an **alternative implementation using Ultralytics YOLO models** for detection, tracking, and pose estimation.

### Primary Optimization Goal

Increase the processing speed to **50 FPS (20ms total time per frame)** on the lab PC (RTX 4060) for potential real-time applications.
**Status: Initial 15-20 FPS target met (~26 FPS). New 50 FPS target requires further optimization.**

### Current Performance Metrics (Post-Optimizations)

| Component       | Avg Time (ms) | Target Time (ms) | Status                 | Notes                                      |
|-----------------|---------------|------------------|------------------------|--------------------------------------------|
| Frame Capture   | ~5            | < 2              | Needs Improvement (Low Priority) | Default OpenCV backend                   |
| Detection       | ~19           | < 8              | Needs Improvement      | Post-normalization optimization          |
| Tracking        | ~1            | < 1              | Optimal                | ByteTrack                                  |
| Pose Estimation | **~11**       | < 7              | Needs Improvement      | **Batch processing implemented**           |
| Display/Storage | ~2            | < 2              | Acceptable             | HDF5 write + OpenCV display              |
| **Total**       | **~38**       | **< 20**         | **Needs Improvement (Target: 50 FPS)** | Current ~26 FPS                          |

## Current Investigation

### RT-DETR Preprocessing Optimization (Latest Work)

Successfully optimized RT-DETR preprocessing using the same cv2-based approach that worked for RTMDet:

**Performance Improvements:**
- RT-DETR preprocessing: **10.5ms → 4.3ms** (59% faster, 2.4x speedup)
- RT-DETR total detection: **30.5ms → 23.7ms** (22% faster overall)
- Overall pipeline: **~52-56ms per frame** (~18-19 FPS)

**Key Changes:**
1. Replaced NumPy normalization with cv2 in-place operations
2. Eliminated BGR→RGB conversion (model accepts BGR input)
3. Removed division by 255 (work in 0-255 range)
4. Updated normalization constants to BGR 0-255 format
5. Added `copy=False` flag to avoid memory allocation

**RT-DETR vs RTMDet Comparison (Post-Optimization):**
- **RT-DETR**: ~23.7ms detection (preprocessing: 4.3ms, inference: ~19ms)
- **RTMDet**: ~19ms detection (preprocessing: 4.5ms, inference: ~12.5ms)
- Gap reduced from 2x slower to only 1.25x slower

### Detection Stage Findings (Current)
- **RTMDet**: ~19ms total (preprocessing: 4.5ms, inference: ~12.5ms)
- **RT-DETR**: ~23.7ms total (preprocessing: 4.3ms, inference: ~19ms)
- Both now use optimized cv2-based preprocessing
- RT-DETR's slower inference is inherent to model architecture

### Pose Estimation Stage (RTMPose) Findings (Post-Batching Opt.)
- **Total time**: **~11ms per frame (for the entire batch)**
- **Preprocessing**: ~4ms (Batch preprocessing time)
- **Inference**: ~6ms (Single inference call for the batch)
- **Postprocessing**: ~0.4ms (Batch postprocessing time)
- **Key Change**: Inference is now called only *once* per frame for all detected boxes.

### Key Bottlenecks Identified (Current)
1.  **Detection Stage**: Now the largest single component (~19ms).
2.  **ONNX Runtime operations** potentially being assigned to CPU instead of GPU (Warning persists):
    ```
    [W:onnxruntime:, session_state.cc:1168 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
    ```
3.  **Possible memory transfer inefficiencies** between CPU and GPU (Implicit transfers still occur).

### Optimization Priorities (Revised)
1.  **GPU Accelerated Capture/Preprocessing**: Explore options like PyNvVideoCodec for GPU decoding and/or rewriting preprocessing steps (resize, normalize, transpose) to run on the GPU (e.g., using PyTorch/CuPy) to minimize CPU bottlenecks and CPU<->GPU transfers. (High complexity, uncertain benefit unless preprocessing is also moved to GPU).
2.  **Further Detection Optimization**: Explore if RTMDet preprocessing/postprocessing can be further optimized (Low priority).
3.  **Model Quantization (FP16/INT8)**: Investigate potential performance gains and accuracy trade-offs of using lower-precision models.

## Recent Changes

1. **Completed functional pipeline** that performs:
   - Video frame extraction
   - Player detection using RTMDet or RT-DETR
   - Player tracking using ByteTrack
   - Pose estimation using RTMPose (**now with batch processing**)
   - Output generation (video overlay and HDF5 data)

2. **Optimized Preprocessing**: Implemented OpenCV-based normalization in `RTMDet`, `RTMPose`, and **RT-DETR** preprocessing, significantly reducing preprocessing time through cv2 in-place operations.

3. **Implemented Batch Pose Estimation**: Modified `RTMPose` and `BaseTool` to process all detected bounding boxes in a single batch, reducing pose estimation time to ~11ms.

4. **Achieved Initial Target FPS**: Overall pipeline speed increased from ~5.4 FPS initially, to ~22 FPS after preprocessing optimization, and then to **~26 FPS** after batch pose estimation, meeting the initial 15-20 FPS target. **New target is 50 FPS.**

5. **Established baseline performance metrics** across different GPUs (though specific timings in this doc might refer to 1070Ti tests).

6. **Identified performance bottlenecks** (updated based on latest optimizations).

7. **Created optimization guide** (may need updating).

8. **Implemented detailed profiling** (remains active).

9. **Changed RTMlib handling** (remains active).
10. **Reverted Detection Frequency Reduction**: Restored `scripts/MAIN.py` to run detection every frame due to tracking accuracy issues observed with infrequent detection.
11. **Refactored Performance Profiling**: Cleaned up timing measurements in `scripts/MAIN.py`, added specific timings for CSV writing and final display steps, updated on-screen/CSV/terminal outputs for clarity and consistency (including tab alignment and first-frame exclusion for stats).

12. **Implemented Modular Detector Architecture**: Created a pluggable detector system with consistent interfaces:
    - `pipeline/detector_base.py`: Detector protocol and factory
    - `pipeline/detectors/rtmdet_onnx.py`: RTMDet adapter (preserves current behavior)
    - `pipeline/detectors/rtdetr_onnx.py`: RT-DETR ONNX adapter (ready for testing)
    - Updated `scripts/MAIN.py` to select detector via `DETECTOR = 'rtmdet' | 'rtdetr'` config
    - Maintains identical output format and profiling regardless of detector choice

13. **RT-DETR ONNX Integration**: Successfully integrated RT-DETRv2 ONNX detector with robust coordinate handling:
    - Model outputs confirmed: ['labels', 'boxes', 'scores'] with boxes in xyxy format
    - Added orig_target_sizes input handling (int64, shape (1,2), (H,W)) - set to model input size (640,640) to ensure proper letterbox coordinate output
    - Person class filtering (COCO=0), confidence threshold 0.70 (increased from 0.30), and NMS implemented
    - **Fixed coordinate scaling issues**: RT-DETR outputs letterbox-space coordinates that require reverse-letterbox transformation
    - **Corrected letterbox preprocessing**: Fixed width/height axis confusion in letterbox resize and padding calculations
    - Implemented reverse-letterbox transformation: b' = (b - pad) / scale to convert from model space to original frame space
    - Added first-frame debug logging and validation assertions
    - **Result**: Boxes now properly positioned on players across the full video processing

14. **RT-DETR Preprocessing Optimization**: Applied cv2-based normalization optimization to RT-DETR:
    - Replaced NumPy operations with cv2.subtract() and cv2.divide() in-place operations
    - Eliminated BGR→RGB conversion (model accepts BGR input)
    - Removed division by 255, work directly in 0-255 range
    - Updated normalization constants from RGB 0-1 to BGR 0-255 format
    - Added copy=False flag to avoid memory allocation
    - **Performance gain**: Preprocessing reduced from 10.5ms to 4.3ms (2.4x speedup)
    - **Overall detection**: Reduced from 30.5ms to 23.7ms (22% faster)
    - RT-DETR now competitive with RTMDet (1.25x slower vs 2x slower before)

## Next Steps

With the detection frequency experiment reverted, the next logical steps focus on other optimization avenues to achieve the **50 FPS target**:

1.  **Explore GPU Accelerated Capture/Preprocessing**: Investigate feasibility and potential benefits of using libraries like `PyNvVideoCodec` for decoding and potentially rewriting preprocessing steps to run entirely on the GPU.
2.  **Further Detection Optimization**: Re-examine RTMDet preprocessing/postprocessing for any remaining minor optimization opportunities (Low priority).
3.  **Model Quantization**: Explore using FP16 or INT8 versions of the RTMDet/RTMPose models if available, assessing performance vs. accuracy.

## YOLO Implementation Exploration

As an alternative to the RTMDet/RTMPose pipeline, we've been exploring an implementation using Ultralytics YOLO models:

### Implementation Status
- Created `scripts/MAIN_YOLO.py` that follows the same pipeline structure as `scripts/MAIN.py`
- Implemented detection using YOLOv8/YOLO11 models
- Integrated ByteTrack for tracking (same as the RTM pipeline)
- Implemented pose estimation using YOLOv8/YOLO11-pose models
- Added visualization and performance profiling

### Performance Findings
- Initial YOLO implementation is significantly slower (~1.9 FPS) than the RTMDet+RTMPose implementation (~26 FPS)
- Attempted per-person pose estimation approach was even slower (0.7 FPS)
- Reverted to whole-frame pose estimation approach

### GPU Compatibility Issues
- Discovered that the RTX 5070 GPU has CUDA capability sm_120
- Current PyTorch builds only support up to sm_90
- This explains why PyTorch can't properly utilize the GPU
- TensorRT acceleration attempts failed due to this compatibility issue

### Implications
- The newer GPU architecture (Blackwell or very new Ampere/Ada Lovelace) is ahead of current stable PyTorch releases
- This limits our ability to fully leverage GPU acceleration with current PyTorch/CUDA tools
- Options include:
  - Continue with RTMDet/RTMPose (already achieving ~26 FPS)
  - CPU-based YOLO (slower)
  - ONNX Format for potential acceleration
  - PyTorch Nightly Builds that might support sm_120

## Current Questions and Considerations

1.  **GPU Preprocessing Benefit**: How much performance gain can be realistically achieved by moving video decoding and/or preprocessing to the GPU?
2.  **FP16/INT8 Accuracy Impact**: How much does lower-precision quantization affect the accuracy of RTMDet and RTMPose for this specific volleyball task? Are pre-quantized models available?
3.  **ONNX CPU Ops Impact**: Are the operations running on the CPU (as indicated by warnings) actually impacting performance significantly, or are they minor shape/metadata operations as ONNX Runtime suggests? Can they be forced to GPU?
4.  **Hardware Bottlenecks**: With the current ~26 FPS (running detection every frame), how much further can we push performance on the RTX 4060 towards the 50 FPS goal using other techniques?
5.  **YOLO vs. RTM Trade-offs**: Is there any benefit to continuing the YOLO implementation given the current performance gap and GPU compatibility issues?
6.  **PyTorch Compatibility**: When will stable PyTorch releases support newer GPU architectures like the RTX 5070's sm_120?
