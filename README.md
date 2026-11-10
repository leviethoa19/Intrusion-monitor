# Indoor Intrusion Monitor

Đồ án GR2 xây dựng hệ thống phát hiện, theo dõi người và cảnh báo xâm nhập qua camera giám sát trong nhà.

## Phạm vi phiên bản đầu tiên

- Đầu vào là video ghi sẵn.
- Detector chính là YOLO11s.
- Chỉ giữ lớp `person` của COCO.
- Xuất video có bounding box và tệp CSV thống kê detection theo frame.
- Có baseline ByteTrack và demo luật cảnh báo bằng cách phát lại tracks đã lưu; chưa tích hợp cảnh báo vào luồng suy luận trực tiếp.

## Demo cảnh báo theo vùng và thời gian

Chạy `src/replay_intrusion.py` với video và tracking run đã có. Có vùng đa giác, bật/tắt, thời gian xác nhận, nhật ký và ảnh minh chứng. Xem [hướng dẫn và giới hạn](notes/intrusion_demo.md). Đây là replay tracks để kiểm tra luật, chưa phải cảnh báo camera trực tiếp.

## Cài đặt trên Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu cần sử dụng một bản PyTorch/CUDA cụ thể, cài PyTorch phù hợp trước rồi mới cài `requirements.txt`.

## Kiểm tra môi trường

```powershell
python src/check_environment.py
```

## Chạy phát hiện người

```powershell
python src/detect_people.py --source data/raw/test_01.mp4
```

Các tùy chọn thường dùng:

```powershell
python src/detect_people.py `
  --source data/raw/test_01.mp4 `
  --model yolo11s.pt `
  --confidence 0.25 `
  --image-size 640 `
  --device auto
```

Kết quả được ghi vào `outputs/<tên-lần-chạy>/`.

Lần chạy kiểm tra đầu tiên đã hoàn thành bằng lệnh:

```powershell
python src/detect_people.py `
  --source data/samples/people_smoke_test.mp4 `
  --model yolo11s.pt `
  --device cpu `
  --run-name day01_smoke_test_v2
```

Kết quả thực tế và các hạn chế được ghi trong `notes/day_01.md`.

## Tiêu chí hoàn thành Ngày 1

- Môi trường Python chạy được.
- YOLO11s tải và suy luận thành công.
- Video kết quả có bounding box quanh người.
- Có `detections.csv` và `summary.json` để ghi nhận kết quả thử nghiệm.

Trạng thái: đã hoàn thành trên video smoke test. Bước kế tiếp là kiểm tra bằng video camera trong nhà, đặc biệt ở điều kiện thiếu sáng.

## Theo dõi người với ByteTrack

Chạy từ thư mục dự án. Chương trình hiện nhận một video cục bộ:

```powershell
python -m pip install -r requirements.txt
python src/track_people.py `
  --source data/raw/day02_indoor.mp4 `
  --model yolo11s.pt `
  --confidence 0.10 `
  --device cpu `
  --run-name day02_bytetrack_baseline
```

Mỗi lần chạy cần một tên mới để tránh ghi đè kết quả. Có thể bỏ `--run-name` để tự tạo tên theo thời gian.

Cấu hình nằm trong `configs/bytetrack.yaml`. Detector chuyển các detection từ 0.10 cho tracker; ByteTrack dùng các detection từ 0.25 trong bước ghép đầu tiên và để tạo track mới. Những detection lớn hơn 0.10 nhưng dưới 0.25 chỉ có thể hỗ trợ ghép với track đang hoạt động trong bước thứ hai; không tự tạo ID mới. Điều này giúp tận dụng detection yếu khi người bị che khuất mà không cho mọi box điểm thấp trở thành một track.

Đầu ra trong `outputs/<tên-lần-chạy>/`:

- `tracked_people.mp4`: video cùng FPS với nguồn, hiển thị ID, confidence và thời điểm.
- `tracks.csv`: một dòng cho mỗi box có ID, gồm frame, thời điểm, ID, confidence và tọa độ.
- `frames.csv`: thống kê mọi frame, kể cả frame không có track, giúp kiểm tra đoạn 17–19 giây.
- `summary.json`: lưu cấu hình, số frame, số ID, khoảng gián đoạn của từng ID và tốc độ toàn pipeline.

Chương trình dùng `persist=True` để giữ trạng thái tracker khi lần lượt xử lý các frame. Nó chỉ vẽ các box quan sát được có ID; không vẽ box dự đoán khi người mất dấu. `track_buffer: 30` giữ track mất dấu trong bộ nhớ theo cơ chế của ByteTrack; không bảo đảm nối lại sau mọi che khuất. Trong thí nghiệm video 30 FPS này, buffer tương ứng khoảng một giây.

Số ID không đồng nghĩa số người thực tế: một người có thể nhận nhiều ID khi mất dấu hoặc ghép sai. Cần xem video để đánh giá đổi ID, bỏ sót và nhận nhầm. Ngưỡng thấp hỗ trợ khôi phục nhưng không loại bỏ hoàn toàn nhận nhầm.

Kiểm thử:

```powershell
python -m pytest -q
```

## So sánh nhanh YOLOv8s và YOLO11s

Đặt hai trọng số pretrained `yolov8s.pt` và `yolo11s.pt` trong thư mục dự án. Chương trình chỉ dùng trọng số đã có để tránh tải mạng trong khi đo.

```powershell
python src/benchmark_detectors.py --source data/raw/day02_indoor.mp4 --device cpu --samples 60 --repeats 3
```

Video được giải mã trước để lấy các frame rải đều, bổ sung các mốc demo 0, 10, 17, 18, 19 và 40 giây nếu có. Hai model nhận đúng cùng frame, cùng confidence 0.25, image size 640, NMS IoU 0.7, batch 1 và FP32. Mỗi model được warmup trước khi đo, thứ tự chạy được đảo giữa frame và lượt đo. Không thêm tracking vào bài thử này.

Kết quả trong `outputs/benchmark_<thời-gian>/` gồm `report.md`, `summary.json`, `timings.csv`, `detections.csv`, ảnh gốc và ảnh đối chiếu hai model. Tên run tự tạo hoặc có thể đặt bằng `--run-name`; không ghi đè run đã tồn tại.

Thời gian đo là lời gọi `predict()` sau warmup, không gồm đọc/ghi video, vẽ ảnh và tải model. FPS này khác FPS toàn hệ thống tracking. Số box và confidence không chứng minh độ chính xác. Vì video chưa có nhãn chuẩn, bài thử chưa có precision/recall/mAP và chưa đủ để chốt model về chất lượng. Xem ảnh đối chiếu để xác định trường hợp cần gán nhãn và đánh giá tiếp.
