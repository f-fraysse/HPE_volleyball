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
   - **Current performance: ~19ms per frame** (Post-normalization optimization). Detection runs every frame.

3. **Player Tracking**
   - ByteTrack algorithm successfully maintains player IDs
   - Handles occlusions and crossing paths effectively
   - Very efficient performance (~1ms per frame)
   - Consistent ID assignment throughout video sequences

4. **Pose Estimation**
   - RTMPose model successfully estimates player poses
   - Medium-sized model (RTMPose-m) provides good accuracy
   - Correctly identifies keypoints for volleyball-specific poses
   - **Current performance: ~11ms per frame** (Post-batching optimization)

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

2. **RTMlib Integration & Modification** ✅
   - Switched from using RTMlib as an installed package to a local editable copy.
   - Modified RTMlib to include detailed timing measurements.
   - Included the modified RTMlib in the project repository.
   - Updated installation instructions in README.md.
   - Modified `RTMPose` and `BaseTool` for batch pose estimation.

3. **Preprocessing Optimization** ✅
   - Optimized normalization using OpenCV functions in both `RTMDet` and `RTMPose`.
   - Reduced preprocessing time significantly.

4. **Batch Pose Estimation** ✅
   - Implemented batch processing in `RTMPose`.
   - Reduced pose estimation time from ~20ms to ~11ms per frame.

5. **Detection Frequency Reduction Experiment** 🟡 (Reverted)
   - Explored running detection every N frames.
   - Encountered tracking accuracy issues (flickering, lost tracks).
   - Reverted `scripts/MAIN.py` to run detection every frame for robustness.

6. **Profiling Refactoring** ✅
   - Cleaned up timing measurements in `scripts/MAIN.py`.
   - Added specific timings for CSV write and final display steps.
   - Updated on-screen/CSV/terminal outputs for clarity and consistency.
   - Ensured terminal statistics exclude the first frame and use tab alignment.

7. **Implementation of Further Optimization Strategies** 🔄
   - Explore GPU accelerated capture/preprocessing - **Next Priority**
   - Explore Model Quantization (FP16/INT8)
   - Further minor detection optimization (Low priority)

8. **YOLO-based Alternative Implementation** 🟡
   - Created `scripts/MAIN_YOLO.py` with the same pipeline structure
   - Implemented detection using YOLOv8/YOLO11 models
   - Integrated ByteTrack for tracking
   - Implemented pose estimation using YOLOv8/YOLO11-pose models
   - Performance significantly slower than RTMDet/RTMPose (~1.9 FPS vs. ~26 FPS)
   - Encountered GPU compatibility issues with RTX 5070 (CUDA capability sm_120)

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

### 🟢 Working Features

- ✅ Video frame extraction
- ✅ Player detection with RTMDet (every frame)
- ✅ Player tracking with ByteTrack
- ✅ Pose estimation with RTMPose (batch processing)
- ✅ Visual output generation
- ✅ HDF5 data storage
- ✅ Performance profiling (Ongoing)

### 🟡 In Progress

- 🔄 Analyzing profiling results (Ongoing)
- 🔄 Exploring further optimization strategies (GPU Preprocessing, Quantization)

### 🔴 Known Issues

1. **Detection Bottleneck**
   - Detection stage (~19ms) is now the primary bottleneck limiting FPS.

2. **ONNX Runtime Warnings**
   - Warnings about operations potentially being assigned to CPU persist. Impact unclear.

## Performance Metrics

| Metric          | Avg Time (ms) | Target (ms) | Status                 | Notes                                      |
|-----------------|---------------|-------------|------------------------|--------------------------------------------|
| Detection Time  | ~19           | < 8         | Needs Improvement      | Post-normalization optimization          |
| Tracking Time   | ~1            | < 1         | 🟢 Optimal             | ByteTrack                                  |
| Pose Time       | ~11           | < 7         | Needs Improvement      | Batch processing implemented             |
| **Total FPS**   | **~26**       | **50 (20ms)** | **Needs Improvement**  | Baseline after batch pose estimation     |
| GPU Memory      | Not measured  | <8GB        | 🟡 To Be Assessed       |                                            |
| CPU Usage       | Not measured  | <50%        | 🟡 To Be Assessed       |                                            |


## Next Milestone

**Explore Further Optimization Strategies (GPU Preprocessing / Quantization)**

**Previous Milestone:** Profiling Refactoring & Documentation Update ✅
- Refactored performance profiling in `scripts/MAIN.py`.
- Updated `README.md` and Memory Bank (`activeContext.md`, `progress.md`).

**Current Task**: Investigate alternative optimization methods like GPU-accelerated preprocessing or model quantization to reach the 50 FPS target.

**Target Completion**: TBD

**Success Criteria**:
- Identify viable alternative optimization techniques.
- Implement and evaluate the most promising technique(s).
- Achieve further progress towards the 50 FPS target.
