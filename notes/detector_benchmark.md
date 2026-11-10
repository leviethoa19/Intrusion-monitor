# So sánh nhanh YOLOv8s và YOLO11s ngày 22/10/2026

## Cách thực hiện

Chạy `src/benchmark_detectors.py` trên `data/raw/day02_indoor.mp4`, video camera cố định dài 49 giây. Lấy 60 frame rải đều và bổ sung các mốc 0, 10, 17, 18, 19, 40 giây; sau khi loại mốc trùng có 65 frame. Cả hai model nhận đúng cùng ảnh nguồn đã giải mã sẵn.

Cấu hình chung: CPU, FP32, batch 1, image size 640, confidence 0.25, NMS IoU 0.7, chỉ lớp person. Warmup 3 lần/model; đo 3 lượt trên 65 frame, tức 195 lời gọi/model. Thứ tự model được đảo giữa frame và lượt đo để giảm ảnh hưởng thứ tự chạy.

Chỉ đo thời gian lời gọi `predict()` sau warmup, gồm tiền xử lý, suy luận, hậu xử lý và overhead. Không tính tải trọng số, đọc/ghi video, vẽ ảnh, tracking. Không dùng số FPS này để so trực tiếp với FPS toàn pipeline tracking.

## Kết quả lần chạy dùng để báo cáo

Thư mục: `outputs/quick_yolov8s_vs_yolo11s_v2/`.

| Chỉ số | YOLOv8s | YOLO11s |
|---|---:|---:|
| Thời gian predict trung bình (ms/frame) | 73.454 | 66.786 |
| Median (ms/frame) | 72.456 | 65.911 |
| P95 (ms/frame) | 83.009 | 76.515 |
| FPS riêng bước detection | 13.614 | 14.973 |
| Tổng detection trên 65 frame (một lượt) | 411 | 453 |
| Frame mẫu có ít nhất một detection | 65 | 65 |

YOLO11s có thời gian predict trung bình thấp hơn khoảng 9.1% trong lần chạy này. Cả ba lượt đều cho YOLO11s có mean latency thấp hơn. Lượt thử trước khi bỏ tham số API deprecated cũng có xu hướng tương tự (khoảng 8%); không gộp hai lượt thành phép đo chính.

Ảnh đối chiếu ở giây 18 có ví dụ YOLO11s vẽ box quanh người ở phía phải trong khi YOLOv8s không có box ở vùng tương ứng tại frame đó. Đây chỉ là quan sát định tính ở một ảnh, không phải kết luận chất lượng tổng thể.

## Cách nói với giáo viên

“Em đã bổ sung phép thử hai model trên cùng 65 frame của video, cùng CPU, image size và ngưỡng confidence; warmup và đo ba lượt. YOLOv8s trung bình khoảng 73.45 ms, YOLO11s khoảng 66.79 ms cho một lần predict, tức YOLO11s giảm latency khoảng 9% trong phép thử này. Em có thêm ảnh đối chiếu cùng frame để kiểm tra những trường hợp khác biệt. Tuy nhiên video chưa có nhãn, nên em mới có cơ sở về tốc độ, chưa đủ cơ sở khẳng định YOLO11s chính xác hơn. Em sẽ giữ YOLO11s làm ứng viên và bổ sung đánh giá trên dữ liệu có nhãn trước khi chốt.”

## Giới hạn và việc tiếp theo

- 453 box so với 411 box không đồng nghĩa YOLO11s chính xác hơn. Box thêm có thể là người được tìm thêm, nhận nhầm hoặc trùng box.
- Cùng ngưỡng 0.25 là thí nghiệm cùng cấu hình; không có nghĩa confidence của hai model được hiệu chuẩn giống nhau hay mỗi model đã được tối ưu.
- Chưa có ground truth nên chưa tính precision, recall, F1, mAP. Chưa đánh giá chất lượng tracking.
- Lấy mẫu không liên tục nên không kết luận độ ổn định box qua mọi frame hoặc đã giải quyết che khuất 17–19 giây.
- Kết quả tốc độ phụ thuộc phần cứng, chế độ nguồn, tải máy và dữ liệu. Nên chạy lại khi cắm nguồn, hạn chế tác vụ nặng và giữ nguyên cấu hình.
- Bước tiếp theo: gán nhãn box người trên một tập frame đại diện riêng, xác định rõ cách gán nhãn người bị che khuất, sau đó đo precision/recall/mAP và chọn cấu hình phù hợp.

## Chạy lại

```powershell
cd "C:\Users\OS\Documents\GitHub\Intrusion-monitor"
.\.venv\Scripts\python.exe src/benchmark_detectors.py --source data/raw/day02_indoor.mp4 --device cpu --samples 60 --repeats 3
```

Bỏ `--run-name` để tự tạo thư mục mới. `report.md` là báo cáo đọc nhanh; `summary.json` lưu cấu hình, phiên bản và SHA256; `timings.csv` lưu mọi lần đo; `detections.csv` lưu box của lượt đầu; `comparisons/` và `originals/` phục vụ xem và gán nhãn sau này.
