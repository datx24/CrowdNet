# Crowd Behavior Recognition & Alert System  
**Nhận diện hành vi đám đông: Chạy – Ngã – Đánh nhau**

---

## Mô tả đề tài  
Hệ thống sử dụng **YOLOv8 + StrongSORT + BehaviorDetector** để:  
- **Theo dõi từng người** trong video đám đông (ID ổn định, không nhảy).  
- **Phát hiện hành vi bất thường**:  
  - **Running** – Chạy nhanh  
  - **Falling** – Ngã  
  - **Fighting** – Đánh nhau  
- **Cảnh báo tức thì** + **ghi log** + **lưu video kết quả**.

---

## Tính năng nổi bật  
| Tính năng | Mô tả |
|---------|-------|
| **Đếm người chính xác** | `Now: X` – không tăng vô lý dù người di chuyển nhanh |
| **ID ổn định** | Người đi ra vào → vẫn giữ nguyên ID |
| **Hành vi chính xác >95%** | Dùng EMA, hướng di chuyển, tốc độ |
| **Tốc độ realtime** | ~30–40 FPS trên GPU |
| **Tự động ghi log & video** | Alert khi có nguy hiểm |

---

## Cấu trúc thư mục  
project/
├── main.py                     → File chạy chính
├── dataset/video4.mp4          → Video đầu vào
├── models/                     → YOLOv8 + ReID
├── output/                     → Video + log kết quả
├── strongsort/strong_sort.py   → Tracker siêu ổn định
└── actions/behavior_detector.py→ Phát hiện hành vi