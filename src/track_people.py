"""Track people in a local video with YOLO11s and ByteTrack."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Reuse device selection and project-local YOLO settings from the baseline.
from detect_people import PERSON_CLASS_ID, PROJECT_ROOT, choose_device, safe_name

import cv2
from ultralytics import YOLO
from ultralytics.utils import YAML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to one local video")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--confidence", type=float, default=0.10,
                        help="Detector input threshold, not the threshold for starting a track")
    parser.add_argument("--tracker", default=str(PROJECT_ROOT / "configs" / "bytetrack.yaml"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name")
    return parser.parse_args()


def tracked_rows(result, frame_index: int, fps: float) -> list[dict]:
    """Only export boxes carrying an ID; raw detections are not tracks."""
    boxes = result.boxes
    if boxes is None or not boxes.is_track:
        return []
    rows = []
    for box in boxes.cpu():
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append({
            "frame_index": frame_index, "time_seconds": round(frame_index / fps, 6),
            "track_id": int(box.id[0]), "class_id": int(box.cls[0]),
            "confidence": round(float(box.conf[0]), 6),
            "x1": round(x1, 2), "y1": round(y1, 2),
            "x2": round(x2, 2), "y2": round(y2, 2),
        })
    return rows


def update_track_stats(stats: dict, rows: list[dict], frame_index: int) -> None:
    for row in rows:
        track_id = row["track_id"]
        if track_id not in stats:
            stats[track_id] = {
                "track_id": track_id, "first_frame": frame_index,
                "last_frame": frame_index, "observed_frames": 1,
                "max_missing_frames_between_observations": 0,
            }
        else:
            item = stats[track_id]
            item["max_missing_frames_between_observations"] = max(
                item["max_missing_frames_between_observations"],
                frame_index - item["last_frame"] - 1,
            )
            item["last_frame"] = frame_index
            item["observed_frames"] += 1


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    if not source.is_file():
        raise SystemExit(f"Local video not found: {source}")
    if not 0 < args.confidence <= 1:
        raise SystemExit("--confidence must be greater than 0 and at most 1")
    tracker_path = Path(args.tracker).resolve()
    tracker_config = YAML.load(str(tracker_path))
    if tracker_config.get("tracker_type") != "bytetrack":
        raise SystemExit("This experiment requires a ByteTrack configuration")
    low = tracker_config["track_low_thresh"]
    high = tracker_config["track_high_thresh"]
    new = tracker_config["new_track_thresh"]
    if not 0 <= low < high <= new <= 1:
        raise SystemExit("Expected 0 <= track_low_thresh < track_high_thresh <= new_track_thresh <= 1")
    if args.confidence > low:
        raise SystemExit(f"Use --confidence <= {low} to preserve low-score recovery detections")

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise SystemExit("Cannot read video or its FPS")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    run_name = args.run_name or f"{safe_name(args.source)}_track_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.output_root).resolve() / run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        capture.release()
        raise SystemExit(f"Output exists; choose a new --run-name: {run_dir}")
    video_path = run_dir / "tracked_people.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise SystemExit("Cannot open MP4 video writer")

    frame_count = frames_with_tracks = observations = 0
    stats = {}
    tracking_ms = 0.0
    device = choose_device(args.device)
    started = perf_counter()
    try:
        model = YOLO(args.model)
        with (run_dir / "tracks.csv").open("w", encoding="utf-8", newline="") as track_file, \
             (run_dir / "frames.csv").open("w", encoding="utf-8", newline="") as frame_file:
            track_csv = csv.DictWriter(track_file, fieldnames=[
                "frame_index", "time_seconds", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2",
            ])
            frame_csv = csv.DictWriter(frame_file, fieldnames=[
                "frame_index", "time_seconds", "tracked_people", "track_ids",
            ])
            track_csv.writeheader()
            frame_csv.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                tick = perf_counter()
                result = model.track(
                    source=frame, persist=True, tracker=str(tracker_path),
                    classes=[PERSON_CLASS_ID], conf=args.confidence, imgsz=args.image_size,
                    device=device, verbose=False,
                )[0]
                tracking_ms += (perf_counter() - tick) * 1000
                rows = tracked_rows(result, frame_count, fps)
                track_csv.writerows(rows)
                frame_csv.writerow({
                    "frame_index": frame_count, "time_seconds": round(frame_count / fps, 6),
                    "tracked_people": len(rows),
                    "track_ids": ";".join(str(row["track_id"]) for row in rows),
                })
                update_track_stats(stats, rows, frame_count)
                # Draw observed tracks only, not raw detections or predicted
                # boxes while a person is lost.
                for row in rows:
                    x1, y1, x2, y2 = [int(row[k]) for k in ("x1", "y1", "x2", "y2")]
                    color = (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {row['track_id']} person {row['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, f"frame {frame_count} | {frame_count / fps:.2f}s", (12, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                writer.write(frame)
                frames_with_tracks += bool(rows)
                observations += len(rows)
                frame_count += 1
                if frame_count % 150 == 0:
                    print(f"Processed {frame_count}/{expected_frames} frames", flush=True)
    finally:
        capture.release()
        writer.release()
    if not frame_count:
        raise SystemExit("No video frames decoded")

    elapsed = perf_counter() - started
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "model": args.model, "tracker": str(tracker_path),
        "tracker_config": tracker_config, "detector_confidence": args.confidence,
        "image_size": args.image_size, "device": str(device), "source_fps": fps,
        "frames_processed": frame_count, "expected_frames": expected_frames,
        "frames_with_tracks": frames_with_tracks, "tracked_observations": observations,
        "unique_track_ids": len(stats), "track_stats": list(stats.values()),
        "mean_track_call_ms": round(tracking_ms / frame_count, 2),
        "processing_seconds": round(elapsed, 2),
        "end_to_end_fps": round(frame_count / elapsed, 2),
        "output_directory": str(run_dir),
        "note": "Track IDs are not a count of unique people. Missing observations are not interpolated. "
                "Timing includes first-call warmup; end-to-end FPS includes model load and video/CSV I/O.",
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
