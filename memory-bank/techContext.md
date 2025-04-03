# Technical Context: HPE_volleyball

## Technologies Used

### Core Libraries and Frameworks

1. **RTMDet & RTMPose**
   - Part of the OpenMMLab ecosystem
   - State-of-the-art models for object detection and pose estimation
   - Accessed through RTMlib Python package

2. **ByteTrack**
   - State-of-the-art multi-object tracking algorithm
   - Modified version included in the project repository

3. **ONNX Runtime**
   - Cross-platform inference engine for ONNX models
   - Used with CUDA backend for GPU acceleration

4. **OpenCV (cv2)**
   - Used for video I/O, image processing, and visualization
   - Handles frame extraction and output video generation

5. **HDF5 (h5py)**
   - Hierarchical data format for storing structured numerical data
   - Used to store detection, tracking, and pose results

### Programming Languages

- **Python**: Primary development language
- **C++**: Used by underlying libraries (ONNX Runtime, OpenCV, CUDA)

### Model Formats

- **ONNX**: Open Neural Network Exchange format
  - Used for both RTMDet and RTMPose models
  - Enables cross-platform deployment and optimization

## Development Setup

### Environment

- **Conda Environment**: `HPE_volleyball`
- **Python Version**: 3.10
- **IDE**: Visual Studio Code

### Hardware

- **Development Machine**: Home PC with RTX 5070
- **Target Machine**: Lab PC with RTX 4060
- **Previous Testing**: GTX 1070 Ti

### Repository Structure

```
HPE_volleyball/
├── ByteTrack/           # Forked + modified ByteTrack repo (tracking)
├── models/              # model files (.onnx) for RTMPose and RTMDet
├── data/                # Input videos
├── output/
│   ├── h5/              # HDF5 outputs: IDs, bboxes, keypoints, scores
│   └── video/           # Output videos with overlays
├── scripts/             # Custom scripts (main pipeline, helpers)
├── paths.py             # Project-relative path definitions
└── requirements.txt     # Python dependencies
```

### Key Dependencies

- **rtmlib**: Wrapper for RTMDet and RTMPose models
- **onnxruntime-gpu**: GPU-accelerated inference engine
- **h5py**: HDF5 file interface
- **opencv-python**: Computer vision operations
- **numpy**: Numerical operations

## Technical Constraints

### Hardware Constraints

1. **GPU Memory**
   - RTX 4060 (target machine) has 8GB VRAM
   - Must optimize memory usage for model inference

2. **Compute Power**
   - Need to balance model size/accuracy with inference speed
   - Target: Process video faster than real-time (>50 FPS)

### Software Constraints

1. **CUDA Compatibility**
   - CUDA version must be compatible with GPU driver
   - ONNX Runtime must be compatible with CUDA version
   - cuDNN version must be compatible with CUDA

2. **ONNX Runtime Limitations**
   - Some operations may be assigned to CPU instead of GPU
   - Need to investigate and optimize model operations

3. **RTMlib Modifications**
   - Required manual modification to output bbox scores:
   ```python
   # Line 141 in rtmdet.py changed from:
   return final_boxes
   # to:
   return final_boxes, final_scores
   ```

### Performance Constraints

1. **Detection Time**: Currently ~40-110ms per frame
2. **Pose Estimation Time**: Currently ~40-75ms per frame
3. **Total Processing**: ~5.4 FPS (vs. 50 FPS video)

## Dependencies and Installation

### Prerequisites

1. **C++ Build Tools for Visual Studio**
   - Required to build Cython wheels

2. **CUDA Toolkit**
   - Compatible with GPU driver
   - Currently tested with CUDA 12.4 and 12.6

3. **cuDNN**
   - Compatible with CUDA version
   - Currently tested with cuDNN 9.7 and 9.8

### Installation Steps

1. **Create and activate conda environment**
   ```bash
   conda create -n HPE-volleyball python=3.10
   conda activate HPE-volleyball
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install ByteTrack**
   ```bash
   cd ByteTrack
   pip install -e .
   cd ..
   ```

4. **Modify RTMlib**
   - Navigate to RTMlib installation
   - Edit rtmdet.py to return scores

## Development Workflow

1. **Model Selection**
   - Download ONNX models from OpenMMLab Deploee
   - Place in `/models` directory

2. **Data Preparation**
   - Place input videos in `/data` directory

3. **Configuration**
   - Edit configuration section in `scripts/MAIN.py`

4. **Execution**
   - Run `scripts/MAIN.py`

5. **Output Analysis**
   - Check output videos in `/output/video`
   - Analyze data in `/output/h5`

## Performance Profiling

Current focus is on detailed profiling of the inference pipeline to identify bottlenecks:

1. **Preprocessing Time**
   - Image resizing
   - Normalization
   - Data format conversions

2. **ONNX Session Time**
   - Actual model inference time
   - GPU operation time

3. **Postprocessing Time**
   - Decoding model outputs
   - Non-maximum suppression
   - Coordinate transformations

4. **Overhead**
   - Memory transfers between CPU and GPU
   - API call overhead
   - Data structure conversions

## Future Technical Considerations

1. **Automated Pipeline**
   - File system monitoring for new videos
   - Automatic processing trigger

2. **Action Recognition**
   - Temporal modeling of pose sequences
   - Classification of volleyball-specific actions

3. **Deployment**
   - Packaging for non-technical users
   - Simplified installation process
