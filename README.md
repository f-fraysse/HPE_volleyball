# HPE_volleyball

This project combines object detection, multi-object tracking, and pose estimation to analyse volleyball training sessions. It uses a customized version of [ByteTrack](https://github.com/ifzhang/ByteTrack) and RTMPose (through [RTMlib](https://github.com/Tau-J/rtmlib)) for tracking and pose analysis of players during spiking actions.

👉 You can download pre-trained RTMDet and RTMPose ONNX models from [OpenMMLab Deploee](https://platform.openmmlab.com/deploee)

🔺 Still in very early stages! 🔺

💹 Detection with RTMdet, tracking with Bytetrack, pose estimation with RTMPose<br>
💹 Save output video with bboxes and poses overlay<br>
❌ Edit tracked IDs manually (delete unused IDs, "relabel" IDs)<br>
❌ Interpolation / smoothing/ manual editing of keypoints<br>
❌ Spike detection from pose data + some heuristics (to start with)<br>
❌ Performance optimisations

## 🎥 Demo

https://github.com/user-attachments/assets/ff8d3e84-efd6-40c7-bcdf-7e803216a152

## 📁 updates

### 2025/04/05: Profiling Cleanup & Detection Frequency Revert<br>
1.  **Detection Frequency Experiment**: Attempted running detection every N frames (e.g., every 3 frames) to improve performance. However, this significantly degraded tracking accuracy (lost tracks, ID switches) due to the rapid and unpredictable movement of players in volleyball. Reverted to running detection on every frame to maintain tracking robustness.
2.  **Profiling Refactoring**: Cleaned up and enhanced the performance profiling in `scripts/MAIN.py`.
    *   Added distinct timing measurements for all major steps within the main loop (Capture, Detection, Tracking, Pose Estimation, HDF5 Write, Display Prep, CSV Write, Final Draw/Display).
    *   Updated on-screen display to show all component times and total FPS.
    *   Updated CSV logging to include all component times.
    *   Added a new "OVERALL TIMING STATISTICS" section to the terminal output, summarizing min/max/avg/median for each component (excluding the first frame).
    *   Formatted all terminal statistics output for better readability using tab alignment.

### 2025/04/05: Batch Pose Estimation<br>
Implemented batch processing for RTMPose estimation.<br>
Previously, each detected bounding box was processed sequentially (preprocess, inference, postprocess).<br>
Now, all bounding boxes in a frame are preprocessed together, inferred in a single batch call to ONNX Runtime, and postprocessed together.<br>
This significantly reduced the pose estimation time from ~20ms to ~11ms per frame.<br>
Overall pipeline speed increased from ~22 FPS to ~26 FPS, meeting the initial 15-20 FPS target.<br>
Detection (~19ms) is now the primary bottleneck.<br>

### 2025/04/04: optimisations<br>
Running ONNXruntime backend - CUDA Execution Provider, on GTX 1070Ti<br>
RTMDet-m and RTMPose-m<br>
video is 1080p 50FPS<br>
Total processing speed approx. 12 FPS (90ms per frame) which is not usable (videos are 2 hours)<br>
I know I could batch process but also want to investigate potential real time applications<br>

#### 1. Changing backend from ONNX to TensorRT
In theory TRT provides great speedup especially running FP16 models.<br>
Wasted many hours trying this.<br>
The easiest would be not to use RTMlib but instead mmdeploy-runtime with a TRT engine.<br>
However MMdeploy supports TRT 8.x (CUDA 11.8 - CudNN 9.7)<br>
GPUs older than Turing (i.e. before RTX 20) do not have tensor cores so do not benefit from FP16 -> gains from TRT FP32 not so big<br>
GPUs newer than Ada (i.e. after RTX 40) are not supported by TRT 8.x (they need TRT 10.x and CUDA 12.8 which MMdeploy doesn't support)<br>
-> in practice, using MMdeploy-SDK (mmdeploy-runtime) with TRT models is only possible with Turing-Ampere-Ada (20, 30, 40 series)<br>
Let's hope MMDeploy team has time to update/maintain again someday as it would be a shame to see that project deprecate.<br>
We are staying on ONNX for now.<br>

#### 2. profiling the script (capture frame -> detection -> tracking -> pose -> export)
Almost all time is spent in detection and pose (45ms det / 45ms pose)<br>
In both RTMdet and RTMpose, preprocessing the frame before inference (normalisation) is significant time cost<br>
I have rewritten the normalisation step. See [notes here](misc project docs/optimising_preprocessing_normalisation.md)<br>
-> saved approx 8ms off det and 5ms off pose<br>
-> total time for 600 frames went from 47 to 31 seconds (50% speedup)<br>


## 📁 Project Structure

```
HPE_volleyball/
├── ByteTrack/           # Forked + modified ByteTrack repo (tracking)
├── models/              # model files (.onnx) for RTMPose, RTMDet, RT-DETR
├── data/                # Input videos
├── output/
│   ├── h5/              # HDF5 outputs: IDs, bboxes, keypoints, scores
│   └── video/           # Output videos with overlays
├── pipeline/            # Modular pipeline components
│   ├── detector_base.py # Detector interface and factory
│   ├── detectors/       # Detector implementations
│   │   ├── rtmdet_onnx.py   # RTMDet ONNX adapter
│   │   └── rtdetr_onnx.py   # RT-DETR ONNX adapter
│   ├── tracker_adapter.py   # ByteTrack wrapper
│   ├── pose_adapter.py      # RTMPose wrapper
│   └── profiling.py         # Profiling utilities
├── scripts/             # Custom scripts (main pipeline, helpers)
├── paths.py             # Project-relative path definitions
└── requirements.txt     # Python dependencies
```

## 🔧 Prerequisites

To run inference on GPU, make sure the following are properly installed:

- **C++ Build Tools for Visual Studio**: C++ compiler is required to build Cython wheels
- **CUDA Toolkit** (e.g. CUDA 12.x or compatible with your PyTorch version)
- **cuDNN** (compatible with your CUDA version)

1. Check which version of CUDA your GPU driver supports:
  ```bash
   nvidia-smi
   ```
   On the top right you will see "CUDA Version", this is the **most recent** version you can use.

2. Download and install CUDA toolkit with appropriate version
3. Now go check [version compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for compatible cuDNN and ONNX runtime versions
4. Download appropriate cuDNN version (use cuDNN archive if you need an older version)

   - For cuDNN, I find the easiest is to copy / paste the dlls from cuDNN folder directly into CUDA folder.
      - {cudNN install path}/bin/{version} -> copy and paste all dlls to {CUDA install path}/bin<br>
      - same for /include (.h files)<br>
      - same for /lib/x64 (.lib files)<br>
   - Alternatively you can add the three cuDNN folder to system PATH.

5. This repo installs onnxruntime-gpu version 1.20.1 by default (CUDA 12.x - cuDNN 9.x), if not compatible with CUDA / cuDNN, then install the compatible one.

Confirmed to work with CUDA 12.4 + CUDNN 9.7 on GTX 1070 Ti<br>
Confirmed to work with CUDA 12.6 + CUDNN 9.8 on GTX 4060

## ⚙️ Setup

1. **Create a conda environment and activate it**
   ```bash
   conda create -n HPE-volleyball python=3.10
   conda activate HPE-volleyball
   ```

2. **Clone this repo**
   ```bash
   git clone https://github.com/f-fraysse/HPE_volleyball.git
   cd HPE_volleyball
   ```

3. **Set up environment**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install ByteTrack**
   ```bash
   cd ByteTrack
   pip install -e .
   cd ..
   ```
5. **Install RTMlib in development mode**
   ```bash
   # Install the included RTMlib in development mode
   cd rtmlib
   pip install -e .
   cd ..
   ```
   
   This installs the included RTMlib in development mode. The necessary modifications for outputting bbox scores and detailed profiling have already been implemented in this local copy.

6. (Optional) Ensure output folders are created:
   ```python
   from scripts.paths import ensure_output_dirs
   ensure_output_dirs()
   ```

## 🚀 Running the Pipeline

Work in progress — main script(s) will be located in `scripts/`.

### Basic Setup

1. add your input video to /data
2. add your ONNX models to /models :
      - download ONNX models from OpenMMLab Deployee: https://platform.openmmlab.com/deploee
      - RTMDet model for detection
      - RTMPose model for pose estimation
      - M-size models seem to provide a good balance of performance and speed ( RTMDet-m, RTMPose-m)
3. run **scripts/MAIN.py**, the start of the script has config options
4. video file with overlaid bboxes, IDs, bbox scores and poses saved in output/video
5. HDF5 file with tracked IDs, bboxes and scores, keypoints and scores saved in output/h5

### Detector Selection

The pipeline supports pluggable detectors. To switch between detectors:

1. **RTMDet (default)**: Set `DETECTOR = 'rtmdet'` in `scripts/MAIN.py`
   - Uses RTMDet ONNX models from OpenMMLab
   - Current baseline: ~19ms detection time

2. **RT-DETR**: Set `DETECTOR = 'rtdetr'` in `scripts/MAIN.py`
   - Uses RT-DETR ONNX models (must be exported first)
   - See [RT-DETR ONNX Export Guide](misc project docs/rtdetr_onnx_export_guide.md)
   - Potential for better real-time performance than RTMDet

The modular pipeline ensures consistent output format and profiling regardless of detector choice.

## 📦 Dependencies

All Python packages are listed in `requirements.txt`.

GPU inference requires a working CUDA installation compatible with your PyTorch/ONNX versions

## 📄 Notes

- ByteTrack has been modified (e.g. fixed deprecated NumPy types).
- RTMlib has been slightly modified (see Setup, Step 5) to output bbox scores
- All paths are defined relative to the project root via `paths.py`.

## ✏️ Author

Francois Fraysse - UniSA

Thanks and credits to:  
- MMPose project - [https://github.com/open-mmlab/mmpose]
- RTMlib - [https://github.com/Tau-J/rtmlib]
- ByteTrack - [https://github.com/ifzhang/ByteTrack]

### 📚 Licensing

This project is licensed under the [Apache 2.0 License](LICENSE).

It includes:
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (MIT License) – see `ByteTrack/LICENSE`
- [RTMLib](https://github.com/open-mmlab/rtmlib) (Apache 2.0 License)
