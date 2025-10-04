"""
Profiling helpers for HPE_volleyball pipeline.

Utilities to standardize timing collection and reporting.
"""

import time
from typing import Dict


class Timer:
    """
    Simple timer for measuring execution times.
    """

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed

    def get_elapsed(self) -> float:
        return self.elapsed


def create_timing_dict(total: float = 0.0, preprocess: float = 0.0,
                      prep: float = 0.0, model: float = 0.0,
                      postprocess: float = 0.0) -> Dict[str, float]:
    """
    Create a standardized timing dictionary.

    Args:
        total: Total time
        preprocess: Preprocessing time
        prep: Data preparation/transfer time
        model: Model inference time
        postprocess: Postprocessing time

    Returns:
        Timing dictionary with required keys
    """
    return {
        'total': total,
        'preprocess': preprocess,
        'prep': prep,
        'model': model,
        'postprocess': postprocess
    }


def combine_timings(*timings: Dict[str, float]) -> Dict[str, float]:
    """
    Combine multiple timing dictionaries by summing values.

    Args:
        *timings: Timing dictionaries to combine

    Returns:
        Combined timing dictionary
    """
    combined = create_timing_dict()
    for timing in timings:
        for key in combined:
            combined[key] += timing.get(key, 0.0)
    return combined
