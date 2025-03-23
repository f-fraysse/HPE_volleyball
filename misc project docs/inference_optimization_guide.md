# 🏃‍♂️ HPE_volleyball Inference Optimization Guide

This document summarizes performance profiling and actionable steps to improve per-frame processing speed without sacrificing model quality.

---

## ⏱️ Current Per-frame Timings

| Step        | Time   | Notes |
|-------------|--------|-------|
| Capture     | 6 ms   | ✅ negligible
| Detection   | 110 ms | 🔴 bottleneck #1
| Tracking    | <1 ms  | ✅ super fast (ByteTrack is efficient)
| Pose        | 75 ms  | 🔴 bottleneck #2
| Display     | <1 ms  | ✅ negligible

> **Detection + Pose = ~185 ms/frame** → ~5.4 FPS

---

## 🔍 Can We Improve Performance Without Degrading Model Quality?

Yes — here are several optimization strategies, ranked by difficulty and impact.

---

## ✅ 1. Run Detection Less Often (Easy, Big Win)

Run detection every **N frames** (e.g., every 3–5), and track forward in between.

```python
if frame_idx % DETECTION_INTERVAL == 0:
    detections = detector(frame)
    tracker.reset_with_detections(detections)
else:
    tracker.update(frame_only=True)  # skip detection
```

✅ Big performance gain  
🟡 Slight accuracy tradeoff  
✅ Works with your current models

---

## ✅ 2. Use Pose Estimation Only for Tracked Boxes

Ensure pose estimation is **only run on active tracked boxes**, not all detections or the full frame.

✅ Efficient  
✅ Likely already in place

---

## ✅ 3. Batch Pose Inference (Medium)

Batch all pose inputs into a single ONNX call:

```python
pose_inputs = [crop_and_preprocess(bbox) for bbox in tracked_bboxes]
pose_outputs = pose_model(pose_inputs)
```

✅ Great GPU performance  
🟡 Needs refactor of pose code

---

## ✅ 4. Enable ONNX Runtime Graph Optimization (Easy)

Set this when creating the ONNX Runtime session:

```python
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

✅ Free performance gain

---

## ✅ 5. Try Model Quantization (FP16 or INT8)

Convert RTMPose or RTMDet to **FP16 ONNX** for faster GPU inference.

Tools:
- `onnxruntime-tools`
- `onnxsim`
- `onnxconverter-common`

🟠 Needs validation for accuracy  
🟡 Medium complexity  
✅ Can yield ~2× speedup

---

## 🧠 Summary of Low-Hanging Fruit

| Tactic                           | Effort | Gain     |
|----------------------------------|--------|----------|
| 🔁 Run detection every N frames  | 🟢 Low  | 🟢 Big    |
| 🧠 Batch pose estimation         | 🟡 Med  | 🟢 Big    |
| ⚙️ Optimize ONNX session         | 🟢 Low  | 🟡 Modest |
| 🧊 Use FP16 ONNX models          | 🟠 Med  | 🟢 Big    |

---

## ✅ Ready to Implement?

Let me know if you want help with:

- Skipping detection frames
- Batchifying pose input
- ONNX quantization to FP16
- Benchmarking before/after

