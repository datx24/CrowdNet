# actions/behavior_detector.py
import numpy as np
from collections import deque
import math
import time

class BehaviorDetector:
    """
    BehaviorDetector Pro - Tối ưu tốc độ, độ chính xác, ổn định
    - Dùng EMA cho tốc độ & tỉ lệ khung
    - Kiểm tra fighting thông minh (góc + tốc độ + hướng)
    - Anti-jitter: lọc nhiễu
    - Tự động xóa track cũ
    """

    def __init__(self,
                 max_history=10,
                 run_speed_thresh=35,
                 fall_ratio_drop=0.6,
                 fight_dist=90,
                 fight_speed_ratio=0.6,
                 ema_alpha=0.7,
                 min_frames=3):
        """
        max_history: số frame lưu lịch sử
        run_speed_thresh: ngưỡng tốc độ (pixel/frame)
        fall_ratio_drop: tỉ lệ giảm chiều cao/rộng để coi là ngã
        fight_dist: khoảng cách pixel để nghi ngờ đánh nhau
        fight_speed_ratio: % tốc độ chạy để coi là "va chạm mạnh"
        ema_alpha: hệ số làm mượt (Exponential Moving Average)
        min_frames: cần ít nhất bao nhiêu frame để phát hiện
        """
        self.tracks = {}          # {id: deque[(cx, cy, w, h, timestamp)]}
        self.ema_speed = {}       # {id: tốc độ mượt}
        self.ema_ratio = {}       # {id: tỉ lệ h/w mượt}
        self.last_state = {}      # {id: trạng thái gần nhất}
        self.last_update = {}     # {id: thời gian cập nhật cuối}

        self.max_history = max_history
        self.run_thresh = run_speed_thresh
        self.fall_drop = fall_ratio_drop
        self.fight_dist = fight_dist
        self.fight_speed = fight_speed_ratio
        self.alpha = ema_alpha
        self.min_frames = min_frames
        self.cleanup_interval = 60  # xóa track cũ mỗi 60s
        self.last_cleanup = time.time()

    def _center(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _size(self, box):
        x1, y1, x2, y2 = box
        return x2 - x1, y2 - y1

    def _aspect_ratio(self, w, h):
        return h / w if w > 0 else 1.0

    def _update_ema(self, ema_dict, track_id, value):
        """Cập nhật EMA"""
        if track_id not in ema_dict:
            ema_dict[track_id] = value
        else:
            ema_dict[track_id] = self.alpha * ema_dict[track_id] + (1 - self.alpha) * value
        return ema_dict[track_id]

    def _calc_speed(self, positions):
        """Tính tốc độ trung bình từ 2 frame gần nhất"""
        if len(positions) < 2:
            return 0.0
        (cx1, cy1, _, _, _), (cx2, cy2, _, _, t2) = positions[-2], positions[-1]
        dt = positions[-1][4] - positions[-2][4]
        if dt <= 0: dt = 1/30  # fallback 30fps
        return math.hypot(cx2 - cx1, cy2 - cy1) / dt

    def detect(self, track_id, box):
        """
        Phát hiện hành vi: normal / running / falling / fighting
        """
        x1, y1, x2, y2 = box
        cx, cy = self._center(box)
        w, h = self._size(box)
        timestamp = time.time()

        # === Khởi tạo track ===
        if track_id not in self.tracks:
            self.tracks[track_id] = deque(maxlen=self.max_history)
            self.ema_speed[track_id] = 0.0
            self.ema_ratio[track_id] = self._aspect_ratio(w, h)

        # === Cập nhật lịch sử ===
        self.tracks[track_id].append((cx, cy, w, h, timestamp))
        self.last_update[track_id] = timestamp

        # === Dọn dẹp track cũ (chống memory leak) ===
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_tracks()
            self.last_cleanup = time.time()

        # === Tính toán các chỉ số mượt ===
        speed = self._calc_speed(self.tracks[track_id])
        smooth_speed = self._update_ema(self.ema_speed, track_id, speed)

        ratio = self._aspect_ratio(w, h)
        smooth_ratio = self._update_ema(self.ema_ratio, track_id, ratio)

        # === Phát hiện hành vi ===
        state = "normal"

        # 1. CHẠY (Running)
        if smooth_speed > self.run_thresh:
            state = "running"

        # 2. NGÃ (Falling) - chỉ khi có đủ frame và tỉ lệ giảm đột ngột
        elif len(self.tracks[track_id]) >= self.min_frames:
            prev_ratio = self.ema_ratio[track_id] if len(self.tracks[track_id]) == self.min_frames else \
                        self._aspect_ratio(*self._size(self._bbox_from_center(self.tracks[track_id][-3][:4])))
            if smooth_ratio < self.fall_drop * prev_ratio:
                state = "falling"

        # 3. ĐÁNH NHAU (Fighting) - kiểm tra va chạm + tốc độ cao
        elif self._is_fighting(track_id, cx, cy, smooth_speed):
            state = "fighting"

        # Lưu trạng thái
        self.last_state[track_id] = state
        return state

    def _is_fighting(self, id1, cx1, cy1, speed1):
        """Kiểm tra va chạm có chủ đích (đánh nhau)"""
        for id2, q in self.tracks.items():
            if id1 == id2 or len(q) == 0:
                continue
            cx2, cy2, _, _, _ = q[-1]
            dist = math.hypot(cx1 - cx2, cy1 - cy2)
            if dist > self.fight_dist:
                continue

            # Tính tốc độ của id2
            speed2 = self.ema_speed.get(id2, 0.0)
            if speed1 < self.run_thresh * self.fight_speed and speed2 < self.run_thresh * self.fight_speed:
                continue

            # Kiểm tra hướng di chuyển có về phía nhau không
            if len(self.tracks[id1]) >= 2 and len(q) >= 2:
                dx1 = cx1 - self.tracks[id1][-2][0]
                dy1 = cy1 - self.tracks[id1][-2][1]
                dx2 = cx2 - q[-2][0]
                dy2 = cy2 - q[-2][1]
                dot = dx1 * (cx2 - cx1) + dy1 * (cy2 - cy1)
                if dot > 0:  # cùng hướng hoặc đứng yên
                    return False
            return True
        return False

    def _bbox_from_center(self, center_data):
        """Chuyển (cx,cy,w,h) → (x1,y1,x2,y2)"""
        cx, cy, w, h = center_data[:4]
        return (cx - w//2, cy - h//2, cx + w//2, cy + h//2)

    def _cleanup_old_tracks(self):
        """Xóa track không hoạt động > 10s"""
        now = time.time()
        dead_ids = [tid for tid, t in self.last_update.items() if now - t > 10.0]
        for tid in dead_ids:
            self.tracks.pop(tid, None)
            self.ema_speed.pop(tid, None)
            self.ema_ratio.pop(tid, None)
            self.last_state.pop(tid, None)
            self.last_update.pop(tid, None)

    # === DEBUG: lấy trạng thái hiện tại ===
    def get_debug_info(self, track_id):
        if track_id not in self.tracks:
            return {}
        speed = self.ema_speed.get(track_id, 0)
        ratio = self.ema_ratio.get(track_id, 1.0)
        state = self.last_state.get(track_id, "normal")
        return {"speed": round(speed, 1), "ratio": round(ratio, 2), "state": state}