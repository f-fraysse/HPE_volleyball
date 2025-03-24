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

https://github.com/user-attachments/assets/3a20771c-83d7-40c8-b43a-f9a36d718dc5

## 📁 Project Structure

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

## 🔧 Prerequisites

To run inference on GPU, make sure the following are properly installed:

- **C++ Build Tools for Visual Studio**: C++ compiler is required to build Cython wheels
- **CUDA Toolkit** (e.g. CUDA 12.x or compatible with your PyTorch version)
- **cuDNN** (compatible with your CUDA version)

For cuDNN, I find the easiest is to copy / paste the dlls from cuDNN folder directly into CUDA folder.

{cudNN install path}/bin/{version} -> copy and paste all dlls to {CUDA install path}/bin
same for /include (.h files)
same for /lib/x64 (.lib files)

Alternatively you can add the three cuDNN folder to system PATH.

Confirmed to work with CUDA 12.4 + CUDNN 9.7 on GTX 1070 Ti
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
5. **Manually change a line in RTMlib** (needed to output bbox scores)<br>

   - navigate to {Miniconda install path}\envs\HPE_volleyball\Lib\site-packages\rtmlib\tools\object_detection<br>
   - Edit rtmdet.py:<br>
   - Line 141 (end of postprocess() function) reads:<br>
   ```python
   return final_boxes
   ```
   change it to:<br>
   ```python
   return final_boxes, final_scores
   ```

6. (Optional) Ensure output folders are created:
   ```python
   from scripts.paths import ensure_output_dirs
   ensure_output_dirs()
   ```

## 🚀 Running the Pipeline

Work in progress — main script(s) will be located in `scripts/`.

1. add your input video to /data
2. add your ONNX models to /models :
      - download ONNX models from OpenMMLab Deployee: https://platform.openmmlab.com/deploee 
      - RTMDet model for detection
      - RTMPose model for pose estimation
      - M-size models seem to provide a good balance of performance and speed ( RTMDet-m, RTMPose-m)
3. run scripts/MAIN.py, the start of the script has config options
4. video file with overlaid bboxes, IDs, bbox scores and poses saved in output/video
5. HDF5 file with tracked IDs, bboxes and scores, keypoints and scores saved in output/h5

## 📦 Dependencies

All Python packages are listed in `requirements.txt`.

GPU inference requires a working CUDA installation compatible with your PyTorch/ONNX versions

## 📄 Notes

- ByteTrack has been modified (e.g. fixed deprecated NumPy types).
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
