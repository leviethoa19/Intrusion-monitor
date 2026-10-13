# Ngày 1 — Khởi tạo detector người

Ngày: 07/10/2026

## Mục tiêu

Khởi tạo dự án và chạy YOLO11s trên một video thử, chỉ giữ lớp `person`, sau đó lưu video đã chú thích và dữ liệu detection.

## Yêu cầu hệ thống v0.1

- Một camera cố định hoặc một video ghi sẵn.
- Detector chính: YOLO11s.
- Chỉ xử lý lớp `person`.
- Chưa có tracking, vùng bảo vệ hoặc cảnh báo xâm nhập.
- Đầu ra: video có bounding box, `detections.csv` và `summary.json`.

## Môi trường

- Python: 3.12.14
- PyTorch: 2.14.1
- OpenCV: 5.0.0.93
- Ultralytics: 8.4.174
- CUDA trong môi trường Python: chưa khả dụng (`torch.cuda.is_available() == False`)
- Thiết bị dùng cho lần chạy đầu: CPU
- Không thể đọc thông tin GPU bằng `nvidia-smi` do quyền hệ thống; chưa cấu hình bản PyTorch hỗ trợ CUDA.

## Kết quả

- Video thử nghiệm: `data/samples/people_smoke_test.mp4` (125 frame, 5 giây, 1280 × 720, 25 FPS).
- Mô hình: YOLO11s pretrained trên COCO; ngưỡng confidence 0.25; kích thước inference 640.
- Trạng thái inference: thành công trên toàn bộ 125 frame.
- Tổng số detection người: 2.975; cả 125 frame đều có ít nhất một người.
- Thời gian inference trung bình: khoảng 54,2 ms/frame trên CPU, chưa bao gồm toàn bộ thời gian đọc/ghi video.
- Kiểm thử mã nguồn: 2 test đã chạy thành công.
- Đầu ra: `outputs/day01_smoke_test_v2/people_smoke_test.avi`, `detections.csv` và `summary.json`.

## Nhận xét ban đầu

- Bounding box bám đúng phần lớn cầu thủ, kể cả một số người nhỏ ở xa camera; pipeline đọc video, lọc lớp `person`, ghi video và xuất CSV đã hoạt động.
- Video bóng đá chỉ là smoke test vì có sẵn nhiều người ở nhiều kích thước. Đây không phải dữ liệu đại diện cho camera an ninh trong nhà ban đêm.
- Chưa thể kết luận độ chính xác của hệ thống từ số lượng detection nói trên vì video chưa được gán nhãn chuẩn để tính precision, recall hoặc mAP.
- Chưa có tracking, vùng bảo vệ, trạng thái bật/tắt giám sát hoặc logic cảnh báo theo thời gian.

## Việc tiếp theo

1. Quay hoặc chọn một video trong nhà phù hợp với tình huống xâm nhập ban đêm.
2. Thử YOLO11s với các mức ánh sáng, khoảng cách và che khuất khác nhau; ghi lại ca bỏ sót và phát hiện nhầm.
3. Cấu hình PyTorch CUDA để tận dụng GPU, sau đó đo lại tốc độ.
4. Khi detector ổn định trên video trong nhà, tích hợp ByteTrack để duy trì ID người qua các frame.
