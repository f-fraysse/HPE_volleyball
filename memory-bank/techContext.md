# Technical Context: HPE_volleyball

## Technologies Used

### Core Libraries and Frameworks

1. **RTMDet & RTMPose**
   - Part of the OpenMMLab ecosystem
   - State-of-the-art models for object detection and pose estimation
   - Accessed through RTMlib Python package

2. **Ultralytics YOLO (Alternative Implementation)**
   - YOLOv8/YOLO11 models for object detection and pose estimation
   - Python API for easy integration
   - Explored as an alternative to RTMDet/RTMPose

3. **ByteTrack**
   - State-of-the-art multi-object tracking algorithm
   - Modified version included in the project repository

4. **ONNX Runtime**
   - Cross-platform inference engine for ONNX models
   - Used with CUDA backend for GPU acceleration

5. **OpenCV (cv2)**
   - Used for video I/O, image processing, and visualization
   - Handles frame extraction and output video generation

6. **HDF5 (h5py)**
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
├── rtmlib/              # Modified RTMlib with profiling capabilities
├── models/              # model files (.onnx) for RTMPose and RTMDet
├── data/                # Input videos
├── output/
│   ├── h5/              # HDF5 outputs: IDs, bboxes, keypoints, scores
│   └── video/           # Output videos with overlays
├── profiling_logs/      # CSV logs of detailed timing data
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
   - **RTX 5070 GPU Compatibility Issue**: The development machine's RTX 5070 has CUDA capability sm_120, but current PyTorch builds only support up to sm_90, limiting GPU acceleration options

2. **ONNX Runtime Limitations**
   - Some operations may be assigned to CPU instead of GPU
   - Need to investigate and optimize model operations

3. **RTMlib Modifications**
   - Local copy of RTMlib included in the repository
   - Modified to include detailed profiling capabilities
   - Modified to output bbox scores for tracking
   - Installed in development mode for easy modification
   - Modified `RTMPose` and `BaseTool` for batch pose estimation

4. **YOLO Implementation Limitations**
   - Alternative YOLO-based implementation significantly slower than RTMDet/RTMPose (~1.9 FPS vs. ~26 FPS)
   - TensorRT acceleration attempts failed due to GPU compatibility issues
   - Per-person pose estimation approach was even slower (0.7 FPS) than whole-frame approach

### Performance Constraints

1. **Detection Time**: ~19 ms/frame (Target: < 8ms for 50 FPS goal)
2. **Pose Estimation Time**: ~11 ms/frame (Target: < 7ms for 50 FPS goal)
3. **Total Processing**: ~26 FPS. Target is **50 FPS (20ms/frame)**. Initial 15-20 FPS target met.

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

4. **Install RTMlib**
   ```bash
   cd rtmlib
   pip install -e .
   cd ..
   ```
   - Installs the included RTMlib in development mode
   - Modifications for bbox scores and profiling already implemented

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

Detailed profiling has been implemented to identify bottlenecks in the inference pipeline:

1. **Profiling Implementation**
   - Modified RTMlib to include detailed timing measurements
   - Added timing for preprocessing, inference, and postprocessing
   - Implemented CSV logging of all timing data
   - Added summary statistics output

2. **Profiled Components**
   - **Preprocessing Time**
     - Image resizing
     - Normalization
     - Data format conversions
   - **ONNX Session Time**
     - Actual model inference time
     - GPU operation time
   - **Postprocessing Time**
     - Decoding model outputs
     - Non-maximum suppression
     - Coordinate transformations
   - **Overhead**
     - Memory transfers between CPU and GPU
     - API call overhead
     - Data structure conversions

3. **Data Collection**
   - CSV logs stored in `/profiling_logs/` directory
   - Each run generates a timestamped CSV file
   - Contains per-frame timing data for all components
   - Summary statistics printed at the end of processing

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
