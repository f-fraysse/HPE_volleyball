# 📷 Real-Time Multi-Camera Streaming — Latency and Optimization

You’re using **4x Ubiquiti G3 Flex IP cameras** connected to a router and streaming to your PC. You’ve got pose estimation working, but you’re seeing increasing delays per stream:

- Cam 1: ~2 sec  
- Cam 2: ~4 sec  
- Cam 3: ~6 sec  
- Cam 4: ~8 sec  

This behavior is common with RTSP and default settings. Here’s a deep dive into **why it happens** and how to **fix it**.

---

## 🧠 What's Happening?

### 1. Buffered RTSP Streams

- RTSP streams (e.g., from IP cameras) are **encoded (H.264)** and **buffered** before decoding.
- Libraries like OpenCV, FFmpeg, or VLC often add **2–10 seconds of buffer** to prevent dropped frames.
- If you open the streams **sequentially**, each one starts further behind real time.

**Result**: You get staggered delays — 2s, 4s, 6s, 8s.

---

### 2. Single-threaded Initialization

If each stream is opened one after another, the total startup time accumulates, and each subsequent stream lags further behind.

---

### 3. No Frame Skipping

If you're reading every frame in sequence but can't keep up, you'll **process older and older frames**, increasing latency over time.

---

## 🛠️ Fixes and Optimizations

### ✅ 1. Use Low-Latency Streaming with FFmpeg or GStreamer

**FFmpeg Command Example:**

```bash
ffmpeg -rtsp_transport tcp -fflags nobuffer -flags low_delay \
       -i rtsp://<camera-ip>/stream -f rawvideo ...
```

**OpenCV + GStreamer Example:**

```python
cv2.VideoCapture('gst-launch-1.0 rtspsrc location=rtsp://... latency=0 ! decodebin ! videoconvert ! appsink')
```

Set latency to `0` or `50` ms to minimize buffering.

---

### ✅ 2. Open Streams in Parallel (Multi-threaded Init)

```python
import threading

def open_stream(url, cap_dict, idx):
    cap = cv2.VideoCapture(url)
    cap_dict[idx] = cap

caps = {}
threads = []
urls = [url1, url2, url3, url4]

for i, url in enumerate(urls):
    t = threading.Thread(target=open_stream, args=(url, caps, i))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

This ensures all streams start at nearly the same time, reducing staggered delay.

---

### ✅ 3. Drop Buffered Frames to Stay in Real Time

```python
# Drop older frames before reading the latest
while cap.grab():
    pass
ret, frame = cap.retrieve()
```

You’ll always get the **most recent** frame with this method.

---

### ✅ 4. Lower Resolution / Bitrate

- Set cameras to **720p** instead of 1080p.
- Lower the **frame rate** or **bitrate** in UniFi Protect.
- This reduces:
  - Network usage
  - Decoder workload
  - Memory consumption

---

### ✅ 5. Use Hardware Decoding (Optional)

If you’re using a GPU, try:

- **NVIDIA NVDEC** (with FFmpeg or OpenCV + CUDA)
- **VAAPI** on Linux (for Intel GPUs)

This offloads H.264 decoding from CPU to GPU.

---

### ✅ 6. Use RTSP-Optimized Libraries (Advanced)

For full control over decoding:

- [**PyAV**](https://github.com/PyAV-Org/PyAV) — FFmpeg bindings for Python
- [**GStreamer**](https://gstreamer.freedesktop.org/) — Highly configurable

These give precise control over latency, buffering, frame dropping, etc.

---

## 📊 Summary: What To Do Next

| Task                          | Benefit                          |
|-------------------------------|----------------------------------|
| Multi-threaded camera init    | Reduces staggered delay          |
| Frame skipping (`.grab()`)    | Reduces latency per stream       |
| Use FFmpeg/GStreamer          | Minimizes decoding buffer delay  |
| Lower res/bitrate             | Frees up bandwidth and CPU       |
| Hardware decode (NVDEC)       | Faster decoding on GPU           |

---

## 🧪 Want a Starter Script?

I can build a sample **multi-camera OpenCV script** with:

- Multi-threaded RTSP capture  
- Real-time `.grab()` logic  
- Pose estimation placeholder  

Just let me know! 🚀
