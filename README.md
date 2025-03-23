# HPE_volleyball

This project combines object detection, multi-object tracking, and pose estimation to analyze volleyball training sessions. It uses a customized version of [ByteTrack](https://github.com/ifzhang/ByteTrack) and RTMPose (through [RTMlib](https://github.com/Tau-J/rtmlib)) for tracking and pose analysis of players during spiking actions.

👉 You can download pre-trained RTMDet and RTMPose ONNX models from [OpenMMLab Deploee](https://platform.openmmlab.com/deploee)

❗ Still in very early stages ❗

💹 Detection with RTMdet, trackin with Bytetrack, pose estimation with RTMPose
💹 Save output video with bboxes and poses overlay
❌ Spike detection from pose data + some heuristics (to start with)
❌ Edit tracked IDs manually (delete unused IDs, "relabel" IDs)
❌ Interpolation / smoothing/ manual editing of keypoints
❌ Performance optimisations

## 🎥 Demo

https://github.com/user-attachments/assets/3a20771c-83d7-40c8-b43a-f9a36d718dc5

## 📁 Project Structure

```
HPE_volleyball/
├── ByteTrack/           # Forked + modified ByteTrack repo (tracking)
├── models/              # RTMPose model weights, ONNX files, etc.
├── data/                # Input videos for processing
├── output/
│   ├── h5/              # HDF5 outputs: poses, IDs, bboxes, scores
│   └── video/           # Output videos with overlays
├── scripts/             # Custom scripts (main pipeline, helpers)
├── paths.py             # Project-relative path definitions
└── requirements.txt     # Python dependencies
```

## 🔧 Prerequisites

To run inference on GPU, make sure the following are properly installed:

- **CUDA Toolkit** (e.g. CUDA 11.8 or compatible with your PyTorch version)
- **cuDNN** (compatible with your CUDA version)

## ⚙️ Setup

1. **Clone this repo**
   ```bash
   git clone https://github.com/yourusername/HPE_volleyball.git
   cd HPE_volleyball
   ```

2. **Set up environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install ByteTrack**
   ```bash
   cd ByteTrack
   pip install -e .
   cd ..
   ```

4. (Optional) Ensure output folders are created:
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

Francois Fraysse — [frayssfe@gmail.com]

Thanks and credits to:  
- MMPose project - [https://github.com/open-mmlab/mmpose]
- RTMlib - [https://github.com/Tau-J/rtmlib]
- ByteTrack - [https://github.com/ifzhang/ByteTrack]

### 📚 Licensing

This project is licensed under the [Apache 2.0 License](LICENSE).

It includes:
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (MIT License) – see `ByteTrack/LICENSE`
- [RTMLib](https://github.com/open-mmlab/rtmlib) (Apache 2.0 License)
