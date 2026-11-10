"""Quick controlled YOLOv8s/YOLO11s detection benchmark on one local video.

No ground truth is inferred: detection counts are not accuracy metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from detect_people import PERSON_CLASS_ID, PROJECT_ROOT, choose_device

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/day02_indoor.mp4")
    parser.add_argument("--models", nargs=2, default=["yolov8s.pt", "yolo11s.pt"])
    parser.add_argument("--samples", type=int, default=60, help="Evenly spaced frames plus demo timestamps")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU, identical for both models")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name")
    return parser.parse_args()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_samples(source, samples):
    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or total < 1:
            raise ValueError("Cannot read video metadata")
        landmarks = {int(t * fps) for t in (0, 10, 17, 18, 19, 40) if int(t * fps) < total}
        indices = set(np.linspace(0, total - 1, min(total, samples), dtype=int).tolist()) | landmarks
        frames = []
        # Sequential decoding avoids codec-dependent seek errors.
        for index in range(total):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video decode ended early at frame {index}")
            if index in indices:
                frames.append((index, frame))
        return frames, fps, total, landmarks
    finally:
        capture.release()


def sync(device):
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(rows, observations, total_frames):
    call_ms = np.array([r["call_ms"] for r in rows])
    counts = np.array([len(boxes) for boxes in observations.values()])
    return {
        "timed_calls": len(rows), "mean_call_ms": round(float(call_ms.mean()), 3),
        "median_call_ms": round(float(np.median(call_ms)), 3),
        "p95_call_ms": round(float(np.percentile(call_ms, 95)), 3),
        "detection_call_fps": round(1000 / float(call_ms.mean()), 3),
        "mean_inference_ms": round(float(np.mean([r["inference_ms"] for r in rows])), 3),
        "round_mean_call_ms": [round(float(np.mean([r["call_ms"] for r in rows if r["repeat"] == rep])), 3)
                               for rep in sorted({r["repeat"] for r in rows})],
        "sampled_frames": total_frames, "detections_first_round": int(counts.sum()),
        "frames_with_detections": int((counts > 0).sum()),
        "mean_detections_per_frame": round(float(counts.mean()), 3),
    }


def main():
    args = parse_args()
    if args.samples < 1 or args.repeats < 1 or args.warmup < 1:
        raise SystemExit("samples, repeats and warmup must be positive")
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        raise SystemExit("confidence and iou must be in (0,1]")
    source = Path(args.source).resolve()
    weights = [Path(m).resolve() for m in args.models]
    if not source.is_file() or not all(p.is_file() for p in weights):
        raise SystemExit("Source and both model weight files must exist locally")
    if weights[0] == weights[1]:
        raise SystemExit("Choose two different weight files")
    frames, fps, total, landmarks = read_samples(source, args.samples)
    run_name = args.run_name or f"benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    output = PROJECT_ROOT / "outputs" / run_name
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit("Run directory already exists; choose another name or omit --run-name")
    (output / "comparisons").mkdir()
    (output / "originals").mkdir()

    models = [YOLO(str(p)) for p in weights]
    names = [f"model_{i + 1}_{p.stem}" for i, p in enumerate(weights)]
    device = choose_device(args.device)
    options = dict(classes=[PERSON_CLASS_ID], conf=args.confidence, iou=args.iou,
                   imgsz=args.image_size, device=device, quantize=None, augment=False,
                   rect=True, verbose=False, save=False)
    for name, model in zip(names, models):
        for _ in range(args.warmup):
            model.predict(frames[0][1], **options)
        sync(device)
        print(f"Warmup complete: {name}", flush=True)

    timings = []
    observations = {name: {} for name in names}
    previews = {}
    for repeat in range(args.repeats):
        for position, (index, frame) in enumerate(frames):
            # Both orders are exercised across frames and repeated rounds.
            order = (0, 1) if (position + repeat) % 2 == 0 else (1, 0)
            for model_index in order:
                name = names[model_index]
                sync(device)
                start = perf_counter()
                result = models[model_index].predict(frame, **options)[0]
                sync(device)
                call_ms = (perf_counter() - start) * 1000
                timings.append({"model": name, "repeat": repeat + 1,
                                "frame_index": index, "time_seconds": round(index / fps, 6),
                                "order_in_pair": order.index(model_index) + 1,
                                "call_ms": call_ms, "inference_ms": result.speed["inference"],
                                "detections": len(result.boxes)})
                if repeat == 0:
                    observations[name][index] = result.boxes.cpu().data.tolist()
                    # Annotation and serialization happen outside the timed call.
                    if index in landmarks:
                        image = result.plot(line_width=2, font_size=12)
                        cv2.rectangle(image, (0, 0), (image.shape[1], 38), (30, 30, 30), -1)
                        text = f"{weights[model_index].stem} | t={index/fps:.2f}s | conf={args.confidence} | boxes={len(result.boxes)}"
                        cv2.putText(image, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                        previews[(name, index)] = image
            if (position + 1) % 20 == 0:
                print(f"Round {repeat + 1}/{args.repeats}: {position + 1}/{len(frames)} sampled frames", flush=True)

    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    with (output / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "frame_index", "time_seconds", "x1", "y1", "x2", "y2", "confidence", "class_id"])
        for name in names:
            for index, boxes in observations[name].items():
                for box in boxes:
                    writer.writerow([name, index, round(index / fps, 6), *box])
    for index, frame in frames:
        if index in landmarks:
            pair = np.hstack([previews[(name, index)] for name in names])
            if not cv2.imwrite(str(output / "comparisons" / f"frame_{index:05d}.jpg"), pair):
                raise RuntimeError("Failed to save comparison image")
            if not cv2.imwrite(str(output / "originals" / f"frame_{index:05d}.jpg"), frame):
                raise RuntimeError("Failed to save original frame")

    metrics = {name: summarize([r for r in timings if r["model"] == name],
                               observations[name], len(frames)) for name in names}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "source_sha256": sha256(source), "source_fps": fps, "source_total_frames": total,
        "source_duration_seconds": total / fps, "sample_frame_indices": [i for i, _ in frames],
        "repeats": args.repeats, "warmup_calls_per_model": args.warmup,
        "confidence": args.confidence, "nms_iou": args.iou, "image_size": args.image_size,
        "batch_size": 1, "precision": "FP32", "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "models": {name: {"weights": str(p), "sha256": sha256(p),
                          "parameters": sum(v.numel() for v in model.model.parameters()),
                          "weight_size_mb": round(p.stat().st_size / (1024**2), 2)}
                   for name, p, model in zip(names, weights, models)},
        "metrics": metrics,
        "timing_scope": "Synchronized predict() call: preprocessing + inference + postprocessing/API overhead. "
                        "Excludes loading weights, warmup, video decoding, plotting, disk output and tracking. "
                        "FPS is reciprocal mean call time, not full video end-to-end FPS.",
        "limitations": "One unlabelled video, sparse sampled frames, no precision/recall/mAP or tracking quality. "
                        "Counts and confidence do not establish accuracy; common confidence is not per-model tuning.",
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# So sánh nhanh YOLOv8s và YOLO11s", "",
             f"Video: `{source.name}`; {len(frames)} frame được lấy trên toàn video và các mốc demo; "
             f"{args.repeats} lượt đo/model, khởi động {args.warmup} lần/model.", "",
             f"Thiết bị: {device}; FP32; batch 1; image size {args.image_size}; "
             f"confidence {args.confidence}; NMS IoU {args.iou}. Hai model chạy luân phiên, cùng frame.", "",
             f"| Chỉ số | {weights[0].stem} (model 1) | {weights[1].stem} (model 2) |",
             "|---|---:|---:|"]
    for label, key in [("Thời gian predict trung bình (ms)", "mean_call_ms"),
                       ("Median (ms)", "median_call_ms"), ("P95 (ms)", "p95_call_ms"),
                       ("FPS riêng bước detection", "detection_call_fps"),
                       ("Inference-only trung bình (ms)", "mean_inference_ms"),
                       ("Tổng box trên các frame mẫu (một lượt)", "detections_first_round"),
                       ("Frame mẫu có ít nhất một box", "frames_with_detections")]:
        lines.append(f"| {label} | {metrics[names[0]][key]} | {metrics[names[1]][key]} |")
    means = [metrics[n]["mean_call_ms"] for n in names]
    winner = int(np.argmin(means))
    reduction = (1 - min(means) / max(means)) * 100
    lines += ["", f"Trong lần đo này, `{weights[winner].stem}` có thời gian predict trung bình thấp hơn "
              f"khoảng {reduction:.1f}% so với model còn lại. Đây là kết quả trên cấu hình và video này.", "",
              "## Cách đọc và giới hạn", "",
              "- Thời gian đo gồm tiền xử lý, suy luận, hậu xử lý và overhead lời gọi predict; không gồm đọc/ghi video, "
              "vẽ box, warmup, tải model hoặc tracking. Không so trực tiếp FPS này với 12.84 FPS của pipeline tracking.",
              "- Box nhiều hơn không có nghĩa chính xác hơn. Frame có box không có nghĩa phát hiện đủ mọi người.",
              "- Video chưa có nhãn chuẩn, nên chưa có precision, recall hoặc mAP. Không đủ bằng chứng kết luận model nào phát hiện tốt hơn.",
              "- Confidence hai model không được hiệu chuẩn giống nhau; đây là đối chiếu cùng cấu hình khởi đầu.",
              "- Xem ảnh cùng frame để ghi nhận bỏ sót, nhận nhầm và box trùng. Muốn lựa chọn model về chất lượng cần gán nhãn một tập kiểm thử riêng.",
              "- Đây là detection trên các frame lấy mẫu, không kiểm tra độ liên tục tracking hoặc mọi frame trong đoạn 17–19 giây.", "",
              "## Các lượt đo", ""]
    for name in names:
        lines.append(f"- {name}: trung bình từng lượt (ms) = {metrics[name]['round_mean_call_ms']}.")
    lines += ["", "## Ảnh đối chiếu", "", "Bên trái: model 1. Bên phải: model 2. Ảnh gốc trong `originals/` để kiểm tra độc lập.", ""]
    for index in sorted(landmarks):
        lines += [f"### Thời điểm {index/fps:.2f} giây", "",
                  f"![So sánh tại frame {index}](comparisons/frame_{index:05d}.jpg)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
