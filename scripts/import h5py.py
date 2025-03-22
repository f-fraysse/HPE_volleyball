import h5py
import numpy as np
import os

# === Path to your saved HDF5 file ===
h5_file = 'test_det-M_pose-M_track-0508.h5'
h5_path = os.path.join("D:\\PythonProjects\\HPE_volleyball\\output\\h5",h5_file)

with h5py.File(h5_path, "r") as f:
    frame_keys = sorted([k for k in f.keys() if k.startswith("frame_")])
    num_frames = len(frame_keys)

    # Track ID index group (if present)
    track_presence = f.get("track_presence", {})

    all_ids = list(track_presence.keys())
    all_ids_int = [int(tid) for tid in all_ids]
    num_ids = len(all_ids_int)

    print(f"📂 File: {h5_path}")
    print(f"🧠 Total frames: {num_frames}")
    print(f"👤 Total unique track IDs: {num_ids}")

    # Optional: print how many frames each ID appears in (first few only)
    print("\n📊 Track presence summary (first 10 IDs):")
    for tid in sorted(all_ids_int)[:10]:
        frames = f["track_presence"][str(tid)][()]
        print(f"  ID {tid:>3}: {len(frames)} frames")

    # Optional: print keypoint stats from first frame
    sample_key = frame_keys[0]
    keypoints = f[sample_key]["keypoints"][()]
    print
