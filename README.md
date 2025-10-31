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

## 🎬 Demo Video
👉 [Xem video demo trên YouTube](https://www.youtube.com/watch?v=lhzehACZYos)
