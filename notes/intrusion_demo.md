# Demo logic cảnh báo theo vùng và thời gian

Ngày bắt đầu tuần 1: 13/10/2026. Ngày 04/11/2026 thuộc tuần 4 (03–09/11).
Tuần theo lịch khác mức hoàn thành công việc: baseline detection/tracking đã có nhưng chưa hoàn tất đánh giá chất lượng.

## Phạm vi và quy tắc

Bản này phát lại video cùng `tracks.csv` đã có, không tải model, không suy luận lại và chưa chạy camera trực tiếp. Mục tiêu là kiểm tra quyết định cảnh báo khi thay chế độ, vùng và thời gian, giữ nguyên đầu ra tracker.

- Điểm đại diện là giữa cạnh dưới box, xấp xỉ vị trí chân; có thể sai khi chân bị che.
- Vùng đa giác dùng tọa độ chuẩn hóa [0, 1], gốc trên trái; điểm trên biên tính là trong vùng. Vùng mẫu phía phải ảnh chỉ để thử chức năng, chưa phải yêu cầu hộ gia đình.
- DISARMED: không cảnh báo, xóa thời gian chờ cũ.
- ARMED + ngoài vùng: OUTSIDE. ARMED + trong vùng: PENDING.
- Cùng ID được quan sát trong vùng ở mọi frame liên tiếp đủ 2 giây theo thời gian video: ALERT, ghi một sự kiện nghi ngờ xâm nhập. Tiếp tục ở vùng không lặp sự kiện trong lượt đó.
- Ra vùng, mất quan sát hoặc tắt giám sát: hủy lượt hiện tại, lần sau tính lại từ đầu. Đổi ID cũng tính lại. Có thể bỏ sót do phân mảnh ID; mất dấu sau ALERT có thể tạo cảnh báo lặp sau khi xác nhận lại. Chưa chống lặp ở cấp một người thật hoặc toàn vùng.
- Người đã ở trong vùng khi bật giám sát vẫn bắt đầu PENDING; chưa yêu cầu vượt biên/hướng vào.

Không suy luận danh tính hay ý đồ. Chưa có chủ nhà hủy/xác nhận sự kiện, thông báo từ xa hoặc clip riêng từng sự kiện. Video chưa có nhãn nên số sự kiện đầu ra không phải số xâm nhập thật hay thước đo độ chính xác.

## Chạy demo

Từ thư mục project trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -B src/replay_intrusion.py `
  --source data/raw/day02_indoor.mp4 `
  --tracking-run outputs/day02_bytetrack_baseline `
  --config configs/intrusion_demo.json `
  --show
```

Lịch mẫu: tắt [0, 5) giây; bật [5, 15); tắt [15, 25); bật từ 25 giây đến hết. Bấm A đảo chế độ, S trở về lịch, Q dừng. Phím có hiệu lực từ frame tiếp theo, cần focus cửa sổ video. Dừng sớm tạo báo cáo `complete: false`.

Bỏ `--show` để không mở cửa sổ. Đối chứng chỉ đổi chế độ: chạy riêng `--mode disarmed`, rồi `--mode armed`, giữ nguyên các đầu vào khác. `--max-frames 300` giới hạn 10 giây đầu video 30 FPS, không đại diện toàn video. Mỗi lần tự tạo thư mục mới, không ghi đè.

## Đầu ra trong outputs

- `intrusion_demo.mp4`: vùng, ID, trạng thái và bộ đếm sự kiện.
- `events.csv` và `event_*.jpg`: thời điểm xác nhận, ID, thời điểm bắt đầu chờ, ảnh minh chứng.
- `states.csv`: mọi frame, chế độ, nguồn điều khiển, ID chờ/cảnh báo, số sự kiện mới.
- `summary.json`: cấu hình, nguồn tracking, hash video/tracks và phạm vi xử lý. Tốc độ replay không phải tốc độ pipeline AI.

Đầu ra đã được Git ignore. Không đưa video/ảnh riêng tư lên GitHub.

## Tiêu chí kiểm tra

Tắt giám sát hoặc ngoài vùng thì không cảnh báo. Trong vùng chưa đủ thời gian chỉ PENDING; đủ thời gian sinh một sự kiện mỗi lượt liên tục. Tắt giám sát, mất dấu, ra vùng, đổi ID đều phải tính thời gian lại. Test tổng hợp kiểm tra luật; cần video có nhãn để đánh giá cảnh báo đúng/nhầm, bỏ sót và độ trễ thực tế.

## Kết quả kiểm tra local ngày 04/11/2026

Nguồn `day02_indoor.mp4`, 1280 × 720, 30 FPS, 49 giây, 1.470 frame. Dùng nguyên CSV của `day02_bytetrack_baseline` (YOLO11s + ByteTrack, CPU, confidence detector 0.10, kích thước 640). Bản replay chỉ giải mã/vẽ và áp luật, không suy luận model; môi trường Python 3.12.14, OpenCV 5.0.0.93 trên Windows. Chưa xác định được tên CPU. Không dùng tốc độ replay làm kết quả benchmark AI.

Giữ nguyên vùng mẫu và ngưỡng 2 giây, chỉ đổi chế độ:

| Chế độ | Frame | Sự kiện theo luật | Thư mục trong outputs |
|---|---:|---:|---|
| Tắt toàn bộ | 1470 | 0 | intrusion_disarmed_full_20261104_01 |
| Bật toàn bộ | 1470 | 7 | intrusion_armed_full_20261104_01 |
| Theo lịch mẫu | 1470 | 6 | intrusion_schedule_full_20261104_01 |

Đã kiểm tra video đầu ra mở được, có đúng 1.470 frame; các sự kiện chỉ xuất hiện lúc armed, có ảnh minh chứng và chuỗi quan sát trong vùng đủ thời gian. Đã xem ảnh sự kiện đầu để kiểm tra vùng/ID/trạng thái. Chưa kiểm tra thao tác phím ở cửa sổ tương tác hoặc xem bằng mắt toàn video.

Hai kiểm tra ngắn 300 frame ban đầu (tắt/bật) đều cho 0 sự kiện; không có lượt quan sát trong vùng đủ 2 giây ở đoạn này. Không điều chỉnh ngưỡng để ép sinh sự kiện. Video nguồn nhìn thấy là cơ sở kinh doanh, chưa đại diện tình huống hộ gia đình. Đây là kiểm tra chức năng, không phải đánh giá precision/recall. Lịch mẫu có ID 469 phát sinh hai sự kiện, cần phân tích mất dấu/ra vùng trước khi kết luận chất lượng chống lặp.
