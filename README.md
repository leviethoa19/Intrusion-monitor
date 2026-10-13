# Indoor Intrusion Monitor

Đồ án GR2 xây dựng hệ thống phát hiện, theo dõi người và cảnh báo xâm nhập qua camera giám sát trong nhà.

## Phạm vi phiên bản đầu tiên

- Đầu vào là video ghi sẵn.
- Detector chính là YOLO11s.
- Chỉ giữ lớp `person` của COCO.
- Xuất video có bounding box và tệp CSV thống kê detection theo frame.
- Tracking, vùng giám sát và cảnh báo xâm nhập sẽ được bổ sung sau khi detector hoạt động ổn định.

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
