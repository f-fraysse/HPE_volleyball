
# Profiling Timing – Base from rtmlib

Using **RTMDet** detector and **RTMPose** pose estimator from `rtmlib`.

| Detection | Pose       | Backend       |
|-----------|------------|---------------|
| RTMDET-M  | RTMPOSE-M  | CUDA ONNX-RT  |

**GPU:** 1070Ti

Initial observation:
> When using the YOLOX detector instead of RTMDet, the preprocessing time is minimal (<1ms) compared to RTMDet (12ms, see below).  
> This suggests a step in RTMDet normalisation is time-consuming.  
> The only difference with YOLOX is normalisation: `(image – mean) / std` at the end of preprocessing.

---
## Base config

### ⏱ Total Time: 46.8 seconds

#### DETECTION TIMING STATISTICS
```
total:      min=22.92ms, max=30.44ms, avg=25.63ms, median=25.49ms
preprocess: min=10.47ms, max=14.55ms, avg=12.48ms, median=12.56ms
inference:  min=12.36ms, max=15.79ms, avg=13.07ms, median=12.93ms
postprocess:min=0.07ms,  max=0.19ms,  avg=0.08ms,  median=0.08ms
```

#### POSE ESTIMATION TIMING STATISTICS
```
total:      min=18.69ms, max=40.18ms, avg=26.39ms, median=24.92ms
preprocess: min=1.47ms,  max=2.10ms,  avg=1.60ms,  median=1.59ms
inference:  min=3.04ms,  max=5.76ms,  avg=3.66ms,  median=3.32ms
postprocess:min=0.08ms,  max=0.18ms,  avg=0.09ms,  median=0.08ms
```

---

## Pre-allocating mean and std for normalisation

In `__init__`:
```python
if self.mean is not None:
    self.mean = np.array(self.mean, dtype=np.float32).reshape((1, 1, 3))
    self.std = np.array(self.std, dtype=np.float32).reshape((1, 1, 3))
```

(And remove similar assignment in `__call__()`)

#### ❌ No Gain

**Total time: 41.3 seconds**

#### DETECTION TIMING STATISTICS
```
total:      min=22.94ms, max=32.70ms, avg=27.08ms, median=26.74ms
preprocess: min=10.37ms, max=17.94ms, avg=14.16ms, median=13.97ms
inference:  min=11.77ms, max=16.02ms, avg=12.84ms, median=12.54ms
postprocess:min=0.07ms,  max=0.18ms,  avg=0.08ms,  median=0.08ms
```

---

## ✅ Using In-Place Normalisation (overwrite array)

```python
padded_img = padded_img.astype(np.float32, copy=False)
padded_img -= self.mean
padded_img /= self.std
```

#### ✅ Good Improvement (~5ms)

#### DETECTION TIMING STATISTICS
```
total:      min=18.81ms, max=23.33ms, avg=20.34ms, median=20.25ms
preprocess: min=6.81ms,  max=9.99ms,  avg=7.56ms,  median=7.49ms
inference:  min=11.80ms, max=14.61ms, avg=12.70ms, median=12.70ms
postprocess:min=0.07ms,  max=0.17ms,  avg=0.08ms,  median=0.08ms
```

---

## ⚡ Using OpenCV for Normalisation

```python
padded_img = padded_img.astype(np.float32, copy=False)
cv2.subtract(padded_img, self.mean, dst=padded_img)
cv2.divide(padded_img, self.std, dst=padded_img)
```

Also updated pose estimator (`RTMPose`) with same in-place OpenCV-based normalisation.

---

### ⏱ Total Time: 30.6 seconds

#### DETECTION TIMING STATISTICS
```
total:      min=15.97ms, max=20.34ms, avg=17.03ms, median=16.93ms
preprocess: min=3.98ms,  max=5.96ms,  avg=4.50ms,  median=4.52ms
inference:  min=11.54ms, max=15.09ms, avg=12.45ms, median=12.39ms
postprocess:min=0.07ms,  max=0.16ms,  avg=0.08ms,  median=0.08ms
```

#### POSE ESTIMATION TIMING STATISTICS
```
total:      min=15.22ms, max=29.65ms, avg=19.96ms, median=19.24ms
preprocess: min=0.66ms,  max=1.00ms,  avg=0.68ms,  median=0.67ms
inference:  min=3.03ms,  max=5.10ms,  avg=3.28ms,  median=3.10ms
postprocess:min=0.08ms,  max=0.20ms,  avg=0.08ms,  median=0.08ms
```
