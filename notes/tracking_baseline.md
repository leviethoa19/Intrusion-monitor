# Baseline YOLO11s + ByteTrack — 19/10/2026

## Vì sao thêm tracking?

Thí nghiệm ngày 2 cho thấy hạ ngưỡng confidence giữ được một số box khi người bị che khuất nhưng làm tăng nhận nhầm. Detector xử lý độc lập từng frame. Tracker liên kết detection theo thời gian và gán ID cho các quan sát có khả năng thuộc cùng một người.

`src/detect_people.py` vẫn là mốc detection ban đầu. `src/track_people.py` dùng các hàm hỗ trợ của detector để chọn thiết bị và thiết lập thư mục cấu hình, đồng thời thực hiện vòng lặp tracking riêng bằng OpenCV.

## Các thay đổi chính

1. OpenCV đọc lần lượt từng frame của một video cục bộ, lấy FPS và kích thước nguồn.
2. Gọi `model.track(frame, persist=True, tracker=...)` thay vì `model.predict(...)`. `persist=True` giữ trạng thái giữa các lần gọi trên các frame liên tiếp. Mỗi lần chạy chương trình tạo model mới nên ID không được nối giữa hai video khác nhau.
3. Cấu hình ByteTrack được lưu riêng trong `configs/bytetrack.yaml` để có thể xem, thay đổi và ghi lại trong báo cáo thí nghiệm.
4. Lấy ID từ `box.id`, cùng confidence và tọa độ. Chỉ xuất/vẽ box có ID; detection chưa được tracker xác nhận không tự trở thành một người đang được theo dõi.
5. Ghi CSV ngay theo từng frame, tránh tích lũy tất cả detection trong RAM. `frames.csv` ghi cả frame không có track.
6. Video kết quả giữ FPS và kích thước nguồn. Thời điểm CSV là `frame_index / FPS`, dùng chỉ số frame bắt đầu từ 0.
7. Thư mục output phải mới để tránh ghi đè một thí nghiệm đã có. Camera trực tiếp và nhiều video cùng lúc chưa thuộc phạm vi chương trình này.
8. Thêm dependency `lap` cho bài toán gán ghép giữa detection và track.

## Phân biệt các ngưỡng

| Tham số | Giá trị | Vai trò |
|---|---:|---|
| Detector `--confidence` | 0.10 | Giữ đầu vào điểm thấp cho tracker |
| `track_high_thresh` | 0.25 | Ngưỡng ghép giai đoạn đầu |
| `track_low_thresh` | 0.10 | Cận dưới cho ghép điểm thấp ở giai đoạn sau |
| `new_track_thresh` | 0.25 | Ngưỡng cho khởi tạo track mới |
| `track_buffer` | 30 | Bộ nhớ tạm cho track mất dấu; khoảng 1 giây trong video 30 FPS này |
| `match_thresh` | 0.8 | Ngưỡng chi phí ghép; không phải confidence |

Ví dụ: một người đã có ID, frame tiếp theo chỉ đạt confidence 0.18. Box này bị detector 0.25 loại bỏ nhưng được giữ ở đầu vào 0.10. ByteTrack có thể ghép nó với track đang hoạt động bằng chuyển động dự đoán và mức chồng lấp box. Detection 0.18 không tự khởi tạo ID mới trong cấu hình này.

Đây không phải quy tắc "thấy người một lần thì luôn nhận ra". Khi ghép không thành công hoặc mất dấu lâu, người có thể nhận ID mới. ByteTrack này không dùng nhận dạng khuôn mặt hoặc đặc trưng ngoại hình ReID. Bước ghép điểm thấp hỗ trợ track đang hoạt động; không bảo đảm nối mọi track đã mất dấu.

`track_buffer` chỉ duy trì trạng thái nội bộ, không vẽ box dự đoán vào video trong lúc không có quan sát. Việc giữ ID và việc luôn có box trên màn hình là hai tiêu chí khác nhau.

## Kết quả lần chạy đầu

- Nguồn: `data/raw/day02_indoor.mp4`, 1280 × 720, 30 FPS, 49 giây.
- Output: `outputs/day02_bytetrack_baseline/`.
- Đọc và ghi đầy đủ 1.470 frame.
- 11.017 quan sát có ID, 102 ID khác nhau xuất hiện trong đầu ra.
- Thời gian toàn pipeline khoảng 114,46 giây, tương đương 12,84 FPS trên CPU.
- Thời gian trung bình một lần gọi tracking khoảng 69,55 ms, gồm xử lý model/tracker và lần khởi động đầu tiên. Không so trực tiếp con số này với inference-only của ngày 1.
- 5 bài kiểm thử đạt, gồm xuất ID/thời điểm, loại detection không có ID và ghi khoảng gián đoạn.

Ở vùng người mặc áo sọc ngồi phía trái, ID 225 có đủ 60 quan sát trong khoảng [17, 19) giây, với confidence thấp nhất khoảng 0.1245. Đây là ví dụ tracker tận dụng box dưới 0.25. Chưa xác nhận đây chính là người mà người dùng đã quan sát mất box.

Kiểm tra frame ở giây 18 cũng cho thấy hai box ID 250 và 251 chồng lên cùng người ngồi phía trên. Có nhiều ID ngắn hạn trên toàn video. Vì vậy, 102 ID không phải 102 người thực tế và chưa thể kết luận tracking đã ổn định. Cần kiểm tra trùng box, phân mảnh ID và khả năng gán sai ID trước khi dùng để cảnh báo.

So với detector baseline, lần chạy này đồng thời thêm tracking và giảm ngưỡng đầu vào từ 0.25 xuống 0.10. Chưa có thí nghiệm đối chứng detector 0.10 cùng cấu hình, nên không quy toàn bộ thay đổi chất lượng cho riêng ByteTrack. Cũng chưa có nhãn chuẩn để đo IDF1/HOTA hoặc đếm ID switches chính xác.

## Cách kiểm tra tiếp

1. Xem video có ID, xác định đúng người gặp vấn đề ở giây 17–19.
2. Theo dõi ID trước, trong và sau khi che khuất; ghi lại lúc mất box, đổi ID hoặc có hai box cho cùng người.
3. Thử xử lý trùng detection bằng ngưỡng NMS IoU và thử `new_track_thresh`, mỗi lần đổi một tham số và lưu tên run mới.
4. Nếu cần chứng minh lợi ích riêng của tracking, chạy detector ở cùng ngưỡng 0.10 làm đối chứng.

Tài liệu tham khảo: https://docs.ultralytics.com/modes/track/
