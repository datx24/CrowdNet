# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 21:50:21 2025

@author: ACER
"""

import numpy as np
import math

def extract_features(track):
    """
    track: deque[(cx, cy, w, h, timestamp)]
    return: np.array([...]) vector đặc trưng
    """
    if len(track) < 2:
        return np.zeros(10)

    speeds = []
    ratios = []
    accelerations = []

    for i in range(1, len(track)):
        cx1, cy1, w1, h1, t1 = track[i-1]
        cx2, cy2, w2, h2, t2 = track[i]

        dt = max(t2 - t1, 1e-3)
        v = math.hypot(cx2 - cx1, cy2 - cy1) / dt
        speeds.append(v)

        r1 = h1 / max(w1, 1)
        r2 = h2 / max(w2, 1)
        ratios.append(r2 / max(r1, 1))

    if len(speeds) >= 2:
        accelerations = np.diff(speeds)

    return np.array([
        np.mean(speeds),
        np.std(speeds),
        np.mean(accelerations) if len(accelerations) > 0 else 0,
        np.std(accelerations) if len(accelerations) > 0 else 0,
        np.mean(ratios),
        np.std(ratios),
        speeds[-1],
        ratios[-1],
        len(track),
        1 if np.mean(speeds) > 50 else 0  # feature phụ
    ])
