# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 21:52:51 2025

@author: ACER
"""

import joblib
from actions.behavior_detector import BehaviorDetector
from actions.feature_extractor import extract_features

class HybridBehaviorDetector(BehaviorDetector):
    def __init__(self, ml_model_path="ml_model.pkl", **kwargs):
        super().__init__(**kwargs)
        self.ml_model = joblib.load(ml_model_path)

    def detect(self, track_id, box):
        # Dùng rule-based trước
        base_state = super().detect(track_id, box)

        # Nếu nghi ngờ bất thường thì mới chạy ML
        if base_state in ["running", "falling", "fighting"] and len(self.tracks[track_id]) >= self.min_frames:
            features = extract_features(self.tracks[track_id]).reshape(1, -1)
            try:
                ml_pred = self.ml_model.predict(features)[0]
                # Ưu tiên kết quả ML nếu khác rule-based
                if ml_pred != base_state:
                    base_state = ml_pred
            except Exception:
                pass

        return base_state
