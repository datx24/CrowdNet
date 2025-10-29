# strongsort/strong_sort.py
import numpy as np
import time

class StrongSORT:
    """
    StrongSORT Pro - Không tăng Now vô lý
    - Gộp track trùng (merge duplicate)
    - Tái sử dụng ID chính xác
    - Đếm Now = len(tracks)
    - EMA smoothing
    """

    def __init__(self, max_age=30, n_init=2, alpha=0.8,
                 iou_threshold=0.3, merge_iou_threshold=0.6,
                 reuse_dist=150, reuse_time=5.0):
        self.tracks = {}      # {id: {'bbox': [x1,y1,x2,y2], 'hits': int, 'age': int}}
        self.lost = {}        # {id: (bbox, timestamp)}
        self.next_id = 0
        self.max_age = max_age
        self.n_init = n_init
        self.alpha = alpha
        self.iou_thresh = iou_threshold
        self.merge_iou_thresh = merge_iou_threshold
        self.reuse_dist = reuse_dist
        self.reuse_time = reuse_time

    @staticmethod
    def iou(bb1, bb2):
        x1, y1, x2, y2 = bb1
        x1_, y1_, x2_, y2_ = bb2
        xi1 = max(x1, x1_); yi1 = max(y1, y1_)
        xi2 = min(x2, x2_); yi2 = min(y2, y2_)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)
        return inter / (area1 + area2 - inter + 1e-6) if (area1 + area2 - inter) > 0 else 0

    @staticmethod
    def center(bb):
        return (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2

    def update(self, detections, frame=None):
        dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        current_time = time.time()
        updated_tracks = []

        unmatched_dets = list(range(len(dets)))
        track_ids = list(self.tracks.keys())

        # === 1. MATCH TRACKS ===
        for tid in track_ids:
            if tid not in self.tracks:
                continue
            track = self.tracks[tid]
            best_iou = 0
            best_idx = -1
            for idx in unmatched_dets:
                iou_val = self.iou(track['bbox'], dets[idx, :4].astype(int).tolist())
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx
            if best_iou > self.iou_thresh and best_idx != -1:
                new_bbox = dets[best_idx, :4].astype(int).tolist()
                track['bbox'] = [
                    int(self.alpha * old + (1 - self.alpha) * new)
                    for old, new in zip(track['bbox'], new_bbox)
                ]
                track['hits'] += 1
                track['age'] = 0
                unmatched_dets.remove(best_idx)
                if track['hits'] >= self.n_init:
                    updated_tracks.append(track['bbox'] + [tid])
            else:
                track['age'] += 1

        # === 2. GỘP TRACK TRÙNG (QUAN TRỌNG!) ===
        self._merge_duplicates()

        # === 3. TẠO TRACK MỚI ===
        for idx in unmatched_dets:
            bbox = dets[idx, :4].astype(int).tolist()
            reused_id = self._try_reuse_lost(bbox, current_time)
            if reused_id is not None:
                self.tracks[reused_id] = {
                    'bbox': bbox,
                    'hits': self.n_init,
                    'age': 0
                }
                if reused_id in self.lost:
                    del self.lost[reused_id]
                tid = reused_id
            else:
                tid = self.next_id
                self.tracks[tid] = {
                    'bbox': bbox,
                    'hits': 1,
                    'age': 0
                }
                self.next_id += 1
            if self.tracks[tid]['hits'] >= self.n_init:
                updated_tracks.append(bbox + [tid])

        # === 4. XÓA TRACK CŨ ===
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['age'] > self.max_age:
                self.lost[tid] = (self.tracks[tid]['bbox'], current_time)
                del self.tracks[tid]

        # === 5. DỌN LOST CŨ ===
        for tid in list(self.lost.keys()):
            if current_time - self.lost[tid][1] > self.reuse_time:
                del self.lost[tid]

        return np.array(updated_tracks) if updated_tracks else np.empty((0, 5))

    def _merge_duplicates(self):
        ids = list(self.tracks.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                if id1 not in self.tracks or id2 not in self.tracks:
                    continue
                if self.iou(self.tracks[id1]['bbox'], self.tracks[id2]['bbox']) > self.merge_iou_thresh:
                    keep = min(id1, id2)
                    remove = max(id1, id2)
                    self.tracks[keep]['hits'] = max(self.tracks[keep]['hits'], self.tracks[remove]['hits'])
                    del self.tracks[remove]

    def _try_reuse_lost(self, bbox, current_time):
        cx, cy = self.center(bbox)
        for tid, (lost_bbox, t) in self.lost.items():
            if current_time - t > self.reuse_time:
                continue
            cx_l, cy_l = self.center(lost_bbox)
            if np.sqrt((cx - cx_l)**2 + (cy - cy_l)**2) < self.reuse_dist:
                return tid
        return None

    def get_active_count(self):
        """Đếm chính xác số người hiện tại"""
        return len(self.tracks)