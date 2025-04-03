# Active Context: HPE_volleyball

## Current Work Focus

The current focus of the HPE_volleyball project is **performance optimization** of the inference pipeline. While the detection-tracking-pose pipeline is functionally complete and produces accurate results, the processing speed needs significant improvement to make the system more practical for real-world use.

### Primary Optimization Goal

Increase the processing speed from the current ~5.4 FPS to a target of at least 15-20 FPS on the lab PC (RTX 4060), with an ideal goal of matching or exceeding the video recording rate (50 FPS).

### Current Performance Metrics

| Component | Current Time | Target Time |
|-----------|--------------|------------|
| Frame Capture | ~5-6 ms | (Already efficient) |
| Detection | ~31-47 ms | <20 ms |
| Tracking | ~1 ms | (Already efficient) |
| Pose Estimation | ~31-47 ms (for all bboxes) | <20 ms |
| Display/Storage | ~1-5 ms | (Already efficient) |
| **Total** | ~90-190 ms/frame | <50 ms/frame |

## Current Investigation

We have completed detailed profiling of the inference pipeline and identified specific bottlenecks within the detection and pose estimation stages:

### Detection Stage (RTMDet) Findings
- **Total time**: ~31-47ms per frame
- **Preprocessing**: ~15-16ms (significant portion of detection time)
- **Inference**: ~15-31ms (varies significantly)
- **Postprocessing**: Minimal in most frames

### Pose Estimation Stage (RTMPose) Findings
- **Total time**: ~31-47ms per frame (for all bounding boxes)
- **Preprocessing**: ~0-3ms per bounding box (average)
- **Inference**: ~6-9ms per bounding box (average)
- **Postprocessing**: Minimal per bounding box
- **Important insight**: The reported preprocessing, inference, and postprocessing times are averages per bounding box, while the total time is for all bounding boxes (typically 5)

### Key Bottlenecks Identified
1. **Sequential processing of bounding boxes** in pose estimation - each bounding box is processed one at a time
2. **Detection preprocessing overhead** - takes ~15-16ms, a significant portion of detection time
3. **ONNX Runtime operations** potentially being assigned to CPU instead of GPU:
   ```
   [W:onnxruntime:, session_state.cc:1168 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
   ```
4. **Possible memory transfer inefficiencies** between CPU and GPU

### Optimization Priorities
Based on our findings, we've identified the following optimization priorities:
1. **ONNX Runtime Optimization** - Configure session options for better performance
2. **Detection Frequency Reduction** - Run detection less frequently
3. **Preprocessing Optimization** - Improve image preprocessing operations
4. **Memory Transfer Optimization** - Minimize CPU-GPU transfers

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

## Recent Optimization Attempts

We have attempted several optimization strategies with limited success:

### 1. ONNX Runtime Optimization ✅

We implemented enhanced session options for ONNX Runtime:

```python
import onnxruntime as ort

# Create optimized session options
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.enable_cpu_mem_arena = False  # Reduce memory usage
options.enable_mem_pattern = False    # May help with GPU memory fragmentation
options.intra_op_num_threads = 4      # Control CPU thread usage

# Configure provider options for CUDA
provider_options = {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}

session = ort.InferenceSession(
    path_or_bytes=onnx_model,
    sess_options=options,
    providers=[('CUDAExecutionProvider', provider_options), 'CPUExecutionProvider']
)
```

**Results:**
- Very little impact on overall performance
- Some improvements in minimum times but similar average performance
- Higher variability in performance with occasional spikes
- ONNX Runtime warning about Memcpy nodes persisted:
  ```
  [W:onnxruntime: transformer_memcpy.cc:74 onnxruntime::MemcpyTransformer::ApplyImpl] 2 Memcpy nodes are added to the graph torch-jit-export for CUDAExecutionProvider.
  ```
- This indicates some operations are still being executed on CPU rather than GPU

### 2. Buffer Preallocation ✅

We implemented buffer preallocation in preprocessing steps to reduce memory allocations:

- Preallocated buffers for image resizing and preprocessing
- Modified RTMPose to use preallocated buffers for input images
- Attempted to minimize memory allocations during inference

**Results:**
- Minimal performance improvement
- Potential introduction of bugs in the detection pipeline
- No significant reduction in processing time

### 3. Memory Transfer Optimization ✅

We attempted to optimize memory transfers between CPU and GPU:

- Corrected model input size parameters
- Ensured proper memory layout for tensor operations
- Minimized unnecessary data conversions

**Results:**
- No significant performance improvements
- Potential issues with the detection pipeline
- Memory transfer bottlenecks appear to be inherent to the ONNX Runtime implementation

## Next Steps

Based on our findings from these optimization attempts, we need to explore more substantial changes:

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
