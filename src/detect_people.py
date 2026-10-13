"""Run person-only YOLO detection on an image, video, stream, or webcam."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep Ultralytics settings inside the project instead of writing to the user's
# roaming profile. Ultralytics creates its own ``Ultralytics`` subdirectory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

from ultralytics import YOLO


PERSON_CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video/image path, URL, or webcam index such as 0")
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics model name or local weight path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence")
    parser.add_argument("--image-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--output-root", default="outputs", help="Parent directory for run outputs")
    parser.add_argument("--run-name", help="Optional output directory name")
    parser.add_argument("--show", action="store_true", help="Display the annotated stream while processing")
    return parser.parse_args()


def normalized_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def safe_name(value: str) -> str:
    name = Path(value).stem if not value.isdigit() else f"webcam_{value}"
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return name or "source"


def choose_device(requested: str) -> str | int:
    if requested != "auto":
        return int(requested) if requested.isdigit() else requested

    try:
        import torch
    except ImportError:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def box_rows(results: Iterator[Any]) -> Iterator[dict[str, Any]]:
    for frame_index, result in enumerate(results):
        boxes = result.boxes
        if boxes is None:
            continue
        for detection_index, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            yield {
                "frame_index": frame_index,
                "detection_index": detection_index,
                "class_id": int(box.cls[0]),
                "confidence": round(float(box.conf[0]), 6),
                "x1": round(float(xyxy[0]), 2),
                "y1": round(float(xyxy[1]), 2),
                "x2": round(float(xyxy[2]), 2),
                "y2": round(float(xyxy[3]), 2),
            }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")

    source = normalized_source(args.source)
    device = choose_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{safe_name(args.source)}_{timestamp}"
    # Use an absolute project path so Ultralytics does not resolve the run
    # against its global ``runs_dir`` setting outside this repository.
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.predict(
        source=source,
        classes=[PERSON_CLASS_ID],
        conf=args.confidence,
        imgsz=args.image_size,
        device=device,
        save=True,
        save_txt=False,
        show=args.show,
        stream=True,
        project=str(output_root),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    rows = list(box_rows(results))
    csv_path = run_dir / "detections.csv"
    fieldnames = [
        "frame_index",
        "detection_index",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    frames_with_people = len({row["frame_index"] for row in rows})
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "model": args.model,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "device": str(device),
        "detections": len(rows),
        "frames_with_people": frames_with_people,
        "output_directory": str(run_dir.resolve()),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
