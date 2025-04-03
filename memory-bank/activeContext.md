# Active Context: HPE_volleyball

## Current Work Focus

The current focus of the HPE_volleyball project is **performance optimization** of the inference pipeline. While the detection-tracking-pose pipeline is functionally complete and produces accurate results, the processing speed needs significant improvement to make the system more practical for real-world use.

### Primary Optimization Goal

Increase the processing speed from the current ~5.4 FPS to a target of at least 15-20 FPS on the lab PC (RTX 4060), with an ideal goal of matching or exceeding the video recording rate (50 FPS).

### Current Performance Metrics

| Component | Current Time | Target Time |
|-----------|--------------|------------|
| Frame Capture | ~5-6 ms | (Already efficient) |
| Detection | ~40-110 ms | <20 ms |
| Tracking | ~1 ms | (Already efficient) |
| Pose Estimation | ~40-75 ms | <20 ms |
| Display/Storage | ~1-5 ms | (Already efficient) |
| **Total** | ~90-190 ms/frame | <50 ms/frame |

## Current Investigation

The immediate focus is on **detailed profiling** of the inference pipeline to identify the specific bottlenecks within the detection and pose estimation stages. This involves:

1. **Breaking down the inference process** into its constituent parts:
   - Preprocessing (image preparation)
   - ONNX session inference (actual model execution)
   - Postprocessing (interpreting model outputs)
   - Overhead (memory transfers, API calls)

2. **Investigating ONNX Runtime warnings** about operations being assigned to CPU instead of GPU:
   ```
   [W:onnxruntime:, session_state.cc:1168 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
   ```

3. **Identifying "low-hanging fruit"** optimizations in preprocessing and postprocessing steps that may yield significant performance gains without requiring model changes.

## Recent Changes

1. **Completed functional pipeline** that performs:
   - Video frame extraction
   - Player detection using RTMDet
   - Player tracking using ByteTrack
   - Pose estimation using RTMPose
   - Output generation (video overlay and HDF5 data)

2. **Established baseline performance metrics** across different GPUs:
   - RTX 5070 (home PC)
   - RTX 4060 (lab PC)
   - GTX 1070 Ti (previous testing)

3. **Identified performance bottlenecks**:
   - Detection stage (~40-110 ms/frame)
   - Pose estimation stage (~40-75 ms/frame)

4. **Created optimization guide** outlining potential strategies for improving performance.

5. **Implemented detailed profiling**:
   - Modified RTMlib to include detailed timing measurements
   - Added timing for preprocessing, inference, and postprocessing in both detection and pose estimation
   - Implemented CSV logging of all timing data
   - Added summary statistics output

6. **Changed RTMlib handling**:
   - Switched from using RTMlib as an installed package to a local editable copy
   - Included the modified RTMlib in the project repository
   - Updated installation instructions in README.md

## Next Steps

### Immediate Tasks

1. **Implement detailed profiling** within the detection and pose estimation processes:
   - Add timing measurements for preprocessing steps
   - Add timing measurements for ONNX session execution
   - Add timing measurements for postprocessing steps
   - Add timing measurements for memory transfers

2. **Analyze ONNX model operations** to understand which operations are being assigned to CPU vs. GPU and why.

3. **Experiment with ONNX Runtime session options** to optimize execution:
   ```python
   options = ort.SessionOptions()
   options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   ```

### Short-term Optimization Strategies

Once profiling is complete, implement the most promising optimization strategies:

1. **Preprocessing optimizations**:
   - Optimize image resizing operations
   - Minimize data format conversions
   - Ensure efficient memory layout

2. **ONNX Runtime optimizations**:
   - Configure execution providers appropriately
   - Enable graph optimizations
   - Investigate operation placement

3. **Postprocessing optimizations**:
   - Optimize non-maximum suppression
   - Streamline coordinate transformations
   - Minimize unnecessary computations

### Medium-term Optimization Strategies

After implementing and evaluating the immediate optimizations:

1. **Run detection less frequently** (every N frames):
   - Implement frame skipping for detection
   - Rely on tracking for intermediate frames
   - Evaluate accuracy vs. performance tradeoff

2. **Batch pose estimation inputs**:
   - Modify pose estimation to process all crops in a single batch
   - Evaluate performance improvement

3. **Explore model quantization**:
   - Convert models to FP16 precision
   - Evaluate accuracy impact
   - Measure performance improvement

## Current Questions and Considerations

1. **Is the bottleneck in the models themselves or in the pre/post-processing?**
   - Need to determine if the actual ONNX inference is the bottleneck or if it's the surrounding operations

2. **Why is performance similar across different GPU hardware?**
   - Investigate if there's a common bottleneck (e.g., CPU operations, memory transfers)

3. **How much can we optimize without sacrificing accuracy?**
   - Need to establish acceptable accuracy thresholds for any optimization

4. **Are there specific ONNX Runtime configurations that could improve performance?**
   - Explore execution provider options and graph optimizations

5. **Could we benefit from custom CUDA kernels for specific operations?**
   - Evaluate if any operations would benefit from custom implementation
