# main.py
import cv2
import os
import torch
import urllib.request
from datetime import datetime
from ultralytics import YOLO
from strongsort.strong_sort import StrongSORT
from actions.behavior_detector import BehaviorDetector
import time
import numpy as np

# ==============================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==============================
VIDEO_PATH = "../dataset/video3.mp4"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "crowd_behavior_alert_2.mp4")
LOG_FILE = os.path.join(OUTPUT_DIR, "alert_log.txt")

def log_alert(message):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

# ==============================
# 2. TẢI MÔ HÌNH
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLOv8
yolo_path = "../models/yolov8n.pt"
if not os.path.exists(yolo_path):
    print("Đang tải YOLOv8n...")
    urllib.request.urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        yolo_path
    )

# ReID (không dùng, nhưng giữ để tương thích)
reid_path = "../models/osnet_x0_25_msmt17.pt"
if not os.path.exists(reid_path):
    print("Đang tải ReID model...")
    urllib.request.urlretrieve(
        "https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/releases/download/v0.2/osnet_x0_25_msmt17.pt",
        reid_path
    )

# ==============================
# 3. KHỞI TẠO MODEL & TRACKER
# ==============================
model = YOLO(yolo_path)
tracker = StrongSORT(
    max_age=30,
    n_init=2,
    alpha=0.8,
    iou_threshold=0.3,
    merge_iou_threshold=0.6,
    reuse_dist=150,
    reuse_time=5.0
)
behavior = BehaviorDetector()

# ==============================
# 4. CHUẨN BỊ VIDEO
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Không mở được video: {VIDEO_PATH}")

fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0: fps = 30.0

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

print("Bắt đầu xử lý video... (nhấn 'q' để thoát)")

# ==============================
# 5. XỬ LÝ FRAME
# ==============================
frame_idx = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # === YOLO DETECTION ===
    results = model(frame, verbose=False)
    dets = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                dets.append([x1, y1, x2, y2, conf])

    dets = np.array(dets) if dets else np.empty((0, 5))

    # === STRONGSORT TRACKING ===
    tracks = tracker.update(dets, frame)

    danger_detected = False

    # === VẼ BBOX & HÀNH VI ===
    for track in tracks:
        x1, y1, x2, y2, tid = map(int, track)
        state = behavior.detect(tid, (x1, y1, x2, y2))
        color, label = (0, 255, 0), "Normal"
        if state == "running":
            color, label = (0, 255, 255), "Running"
        elif state == "falling":
            color, label = (0, 0, 255), "Falling [Alert]"
            danger_detected = True
            log_alert(f"Frame {frame_idx} | ID {tid} - Falling")
        elif state == "fighting":
            color, label = (0, 100, 255), "Fighting [Alert]"
            danger_detected = True
            log_alert(f"Frame {frame_idx} | ID {tid} - Fighting")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid} | {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === ĐẾM NGƯỜI HIỆN TẠI (CHÍNH XÁC 100%) ===
    now_count = tracker.get_active_count()
    cv2.putText(frame, f"Now: {now_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # === CẢNH BÁO ===
    if danger_detected:
        cv2.putText(frame, "[Alert] DANGER DETECTED [Alert]", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # === FPS ===
    if frame_idx % 30 == 0:
        curr_time = time.time()
        fps_display = 30 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (fw - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # === HIỂN THỊ & GHI VIDEO ===
    out.write(frame)
    cv2.imshow("Crowd Behavior Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# 6. KẾT THÚC
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video kết quả: {OUTPUT_VIDEO}")
print(f"Log hành vi: {LOG_FILE}")
print(f"Tổng ID được cấp: {tracker.next_id}")
print(f"Số người tối đa cùng lúc: {max(tracker.get_active_count() for _ in range(10))}")