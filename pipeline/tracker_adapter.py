"""
Tracker adapter for HPE_volleyball pipeline.

Wrapper around yolox.tracker.BYTETracker to provide a clean interface.
"""

from typing import List
import numpy as np
from argparse import Namespace
from yolox.tracker.byte_tracker import BYTETracker


class Track:
    """
    Simple track representation.
    """
    def __init__(self, track_id: int, bbox: list[float], score: float):
        self.track_id = track_id
        self.tlwh = bbox  # [x, y, w, h]
        self.score = score
        self.is_activated = True  # Assume all returned tracks are active


def create_tracker(track_thresh: float = 0.5, match_thresh: float = 0.8,
                  track_buffer: int = 30, frame_rate: int = 30) -> BYTETracker:
    """
    Create BYTETracker instance with given parameters.

    Args:
        track_thresh: Detection threshold
        match_thresh: Matching threshold
        track_buffer: Track buffer size
        frame_rate: Frame rate

    Returns:
        BYTETracker instance
    """
    args = Namespace(
        track_thresh=track_thresh,
        match_thresh=match_thresh,
        track_buffer=track_buffer,
        frame_rate=frame_rate,
        mot20=False,
        min_hits=3
    )
    return BYTETracker(args)


def update_tracker(tracker: BYTETracker, dets_for_tracker: np.ndarray,
                  img_info: tuple[int, int], img_size: tuple[int, int]) -> List[Track]:
    """
    Update tracker with detections and return tracks.

    Args:
        tracker: BYTETracker instance
        dets_for_tracker: Detections in format [x1, y1, x2, y2, score, class_id]
        img_info: Image height, width
        img_size: Image size (height, width)

    Returns:
        List of Track objects
    """
    # BYTETracker expects dets as numpy array
    tracks_raw = tracker.update(dets_for_tracker, img_info, img_size)

    # Convert to our Track objects
    tracks = []
    for track in tracks_raw:
        # Convert tlwh to list
        bbox = [float(x) for x in track.tlwh]
        score = float(track.score) if hasattr(track, 'score') else 0.0
        tracks.append(Track(int(track.track_id), bbox, score))

    return tracks
