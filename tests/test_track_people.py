"""Check tracking exports using real Ultralytics result objects."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from track_people import tracked_rows, update_track_stats
from ultralytics.engine.results import Results


def make_result(boxes):
    return Results(np.zeros((100, 100, 3), dtype=np.uint8), path="frame.jpg",
                   names={0: "person"}, boxes=torch.as_tensor(boxes, dtype=torch.float32))


def test_export_track_id_and_source_time():
    result = make_result([[10, 20, 30, 90, 7, 0.18, 0]])
    rows = tracked_rows(result, 510, 30)
    assert rows == [{"frame_index": 510, "time_seconds": 17.0, "track_id": 7,
                     "class_id": 0, "confidence": 0.18,
                     "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 90.0}]


def test_untracked_detections_do_not_become_tracks():
    result = make_result([[10, 20, 30, 90, 0.18, 0]])
    assert tracked_rows(result, 0, 30) == []
    assert tracked_rows(make_result(torch.empty((0, 7))), 0, 30) == []


def test_gap_records_missing_frames_without_inventing_observations():
    stats = {}
    update_track_stats(stats, [{"track_id": 7}], 509)
    update_track_stats(stats, [], 510)
    update_track_stats(stats, [{"track_id": 7}], 570)
    assert stats[7]["observed_frames"] == 2
    assert stats[7]["max_missing_frames_between_observations"] == 60
