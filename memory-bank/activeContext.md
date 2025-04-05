# Active Context: HPE_volleyball

## Current Work Focus

The current focus of the HPE_volleyball project is **performance optimization** of the inference pipeline. While the detection-tracking-pose pipeline is functionally complete and produces accurate results, the processing speed needs significant improvement to make the system more practical for real-world use.

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

Focus remains on performance, but the primary bottleneck has shifted slightly.

### Detection Stage (RTMDet) Findings (Post-Normalization Opt.)
- **Total time**: ~17-19ms per frame
- **Preprocessing**: ~4.5ms (Optimized using OpenCV functions)
- **Inference**: ~12.5ms
- **Postprocessing**: Minimal

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
   - Player detection using RTMDet
   - Player tracking using ByteTrack
   - Pose estimation using RTMPose (**now with batch processing**)
   - Output generation (video overlay and HDF5 data)

2. **Optimized Preprocessing**: Implemented OpenCV-based normalization in `RTMDet` and `RTMPose` preprocessing, significantly reducing preprocessing time. (~17ms det, ~20ms pose baseline achieved).

3. **Implemented Batch Pose Estimation**: Modified `RTMPose` and `BaseTool` to process all detected bounding boxes in a single batch, reducing pose estimation time to ~11ms.

4. **Achieved Initial Target FPS**: Overall pipeline speed increased from ~5.4 FPS initially, to ~22 FPS after preprocessing optimization, and then to **~26 FPS** after batch pose estimation, meeting the initial 15-20 FPS target. **New target is 50 FPS.**

5. **Established baseline performance metrics** across different GPUs (though specific timings in this doc might refer to 1070Ti tests).

6. **Identified performance bottlenecks** (updated based on latest optimizations).

7. **Created optimization guide** (may need updating).

8. **Implemented detailed profiling** (remains active).

9. **Changed RTMlib handling** (remains active).
10. **Reverted Detection Frequency Reduction**: Restored `scripts/MAIN.py` to run detection every frame due to tracking accuracy issues observed with infrequent detection.
11. **Refactored Performance Profiling**: Cleaned up timing measurements in `scripts/MAIN.py`, added specific timings for CSV writing and final display steps, updated on-screen/CSV/terminal outputs for clarity and consistency (including tab alignment and first-frame exclusion for stats).

## Next Steps

With the detection frequency experiment reverted, the next logical steps focus on other optimization avenues to achieve the **50 FPS target**:

1.  **Explore GPU Accelerated Capture/Preprocessing**: Investigate feasibility and potential benefits of using libraries like `PyNvVideoCodec` for decoding and potentially rewriting preprocessing steps to run entirely on the GPU.
2.  **Further Detection Optimization**: Re-examine RTMDet preprocessing/postprocessing for any remaining minor optimization opportunities (Low priority).
3.  **Model Quantization**: Explore using FP16 or INT8 versions of the RTMDet/RTMPose models if available, assessing performance vs. accuracy.

## Current Questions and Considerations

1.  **GPU Preprocessing Benefit**: How much performance gain can be realistically achieved by moving video decoding and/or preprocessing to the GPU?
2.  **FP16/INT8 Accuracy Impact**: How much does lower-precision quantization affect the accuracy of RTMDet and RTMPose for this specific volleyball task? Are pre-quantized models available?
3.  **ONNX CPU Ops Impact**: Are the operations running on the CPU (as indicated by warnings) actually impacting performance significantly, or are they minor shape/metadata operations as ONNX Runtime suggests? Can they be forced to GPU?
4.  **Hardware Bottlenecks**: With the current ~26 FPS (running detection every frame), how much further can we push performance on the RTX 4060 towards the 50 FPS goal using other techniques?
