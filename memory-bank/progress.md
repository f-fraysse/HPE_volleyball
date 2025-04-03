# Project Progress: HPE_volleyball

## What Works

### ✅ Core Pipeline

1. **Video Input**
   - Successfully reads video files using OpenCV
   - Handles standard 1920x1080, 50fps volleyball training videos
   - Frame extraction is efficient (~5-6ms per frame)

2. **Player Detection**
   - RTMDet model successfully detects players in frames
   - Medium-sized model (RTMDet-m) provides good accuracy
   - Detection works reliably even with partial occlusions
   - Current performance: ~40-110ms per frame

3. **Player Tracking**
   - ByteTrack algorithm successfully maintains player IDs
   - Handles occlusions and crossing paths effectively
   - Very efficient performance (~1ms per frame)
   - Consistent ID assignment throughout video sequences

4. **Pose Estimation**
   - RTMPose model successfully estimates player poses
   - Medium-sized model (RTMPose-m) provides good accuracy
   - Correctly identifies keypoints for volleyball-specific poses
   - Current performance: ~40-75ms per frame

5. **Output Generation**
   - Visual output with overlaid bounding boxes, IDs, and pose skeletons
   - HDF5 data storage with structured organization
   - Track presence indexing for efficient data retrieval

### ✅ Development Environment

1. **Conda Environment**
   - Successfully configured with all dependencies
   - Python 3.10 environment with required packages
   - CUDA and cuDNN properly installed and configured

2. **Model Management**
   - ONNX models properly loaded and executed
   - GPU acceleration working through ONNX Runtime
   - Model paths properly configured through paths.py

3. **Project Structure**
   - Organized directory structure for models, data, and output
   - Clear separation of concerns in code organization
   - Path management through centralized paths.py

## What's Left to Build

### 🔄 Current Focus: Performance Optimization

1. **Detailed Profiling** ✅
   - Implemented detailed timing measurements for each component
   - Added breakdown of preprocessing, inference, and postprocessing times
   - Added CSV logging of all timing data
   - Added summary statistics output

2. **RTMlib Integration** ✅
   - Switched from using RTMlib as an installed package to a local editable copy
   - Modified RTMlib to include detailed timing measurements
   - Included the modified RTMlib in the project repository
   - Updated installation instructions in README.md

3. **ONNX Runtime Optimization**
   - Investigate and resolve warnings about CPU operations
   - Experiment with graph optimization settings
   - Configure execution providers appropriately

4. **Preprocessing/Postprocessing Optimization**
   - Identify and optimize inefficient operations based on profiling results
   - Minimize unnecessary data conversions
   - Optimize memory usage and transfers

5. **Implementation of Optimization Strategies**
   - Run detection less frequently (every N frames)
   - Batch pose estimation inputs
   - Explore model quantization (FP16)

### 🔜 Future Work: Action Recognition

1. **Temporal Analysis**
   - Develop methods to analyze pose sequences over time
   - Implement sliding window approach for action detection
   - Create temporal features from pose data

2. **Action Classification**
   - Define volleyball-specific actions (spike, serve, block, etc.)
   - Develop classification model for action recognition
   - Train and evaluate on volleyball pose sequences

3. **Action Visualization**
   - Enhance output video with action labels
   - Create timeline visualization of detected actions
   - Generate action summary statistics

### 🔜 Future Work: Automated Pipeline

1. **File System Monitoring**
   - Implement automatic detection of new video files
   - Create trigger mechanism for processing new videos
   - Develop queue management for multiple videos

2. **Background Processing**
   - Run processing as a background service
   - Implement logging and error handling
   - Create notification system for completed processing

3. **Results Dashboard**
   - Develop interface for reviewing processed videos
   - Create visualization tools for pose and action data
   - Implement filtering and search capabilities

## Current Status

### 🟢 Working Features

- ✅ Video frame extraction
- ✅ Player detection with RTMDet
- ✅ Player tracking with ByteTrack
- ✅ Pose estimation with RTMPose
- ✅ Visual output generation
- ✅ HDF5 data storage

### 🟡 In Progress

- ✅ Performance profiling
- ✅ Analyzing profiling results
- ✅ Identifying optimization opportunities
- 🔄 Implementing ONNX Runtime optimizations

### 🔴 Known Issues

1. **Performance Bottlenecks**
   - Overall processing speed (~5.4 FPS) is too slow for real-time analysis
   - Detection and pose estimation stages are the main bottlenecks
   - Similar performance across different GPU hardware suggests potential CPU bottleneck or memory transfer issues

2. **ONNX Runtime Warnings**
   - Warnings about operations being assigned to CPU instead of GPU
   - Potential impact on performance due to CPU-GPU transfers

3. **RTMlib Modification** ✅
   - Addressed by using a local editable copy of RTMlib
   - Modifications for bbox scores and profiling are now part of the project repository
   - No longer dependent on manual modifications to installed packages

## Performance Metrics

| Metric | Current Value | Target Value | Status |
|--------|---------------|--------------|--------|
| Detection Time | 31-47 ms | <20 ms | 🔴 Needs Improvement |
| Tracking Time | ~1 ms | ~1 ms | 🟢 Optimal |
| Pose Time | 31-47 ms (all bboxes) | <20 ms | 🔴 Needs Improvement |
| Total FPS | ~5.4 | >15-20 | 🔴 Needs Improvement |
| GPU Memory | Not measured | <8GB | 🟡 To Be Assessed |
| CPU Usage | Not measured | <50% | 🟡 To Be Assessed |

## Next Milestone

**Implement ONNX Runtime Optimizations**

We have analyzed the profiling results and identified key bottlenecks:

1. **Detection Stage Findings**:
   - Preprocessing: ~15-16ms (significant portion of detection time)
   - Inference: ~15-31ms (varies significantly)
   - Postprocessing: Minimal

2. **Pose Estimation Stage Findings**:
   - Sequential processing of bounding boxes is a major bottleneck
   - Each bounding box takes ~6-9ms for inference
   - With 5 bounding boxes, this adds up to ~31-47ms total

3. **Optimization Priorities**:
   - ONNX Runtime Optimization (current focus)
   - Detection Frequency Reduction
   - Preprocessing Optimization
   - Memory Transfer Optimization

**Current Task**: Implement ONNX Runtime optimization with enhanced session options

**Target Completion**: TBD

**Success Criteria**: 
- Implementation of ONNX Runtime optimizations
- Measurable improvement in inference speed
- Reduction in CPU-GPU transfer overhead
