"""Verify aggregation and video sampling used by the quick benchmark."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark_detectors import read_samples, summarize


def test_summary_uses_all_repeats_but_counts_detections_once():
    rows = [{"call_ms": value, "inference_ms": value / 2, "repeat": repeat}
            for repeat, values in [(1, [10, 20]), (2, [30, 40])]
            for value in values]
    stats = summarize(rows, {0: [[0, 0, 1, 1, .8, 0]], 9: []}, 2)
    assert stats["mean_call_ms"] == 25
    assert stats["detection_call_fps"] == 40
    assert stats["round_mean_call_ms"] == [15, 35]
    assert stats["detections_first_round"] == 1
    assert stats["frames_with_detections"] == 1
    assert stats["p95_call_ms"] == pytest.approx(38.5)


def test_video_sample_indices_are_unique_and_include_demo_landmarks(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (32, 32))
    assert writer.isOpened()
    for index in range(21):
        writer.write(np.full((32, 32, 3), index * 10, dtype=np.uint8))
    writer.release()
    frames, fps, total, landmarks = read_samples(path, 3)
    indices = [index for index, _ in frames]
    assert total == 21 and fps == 1
    assert indices == [0, 10, 17, 18, 19, 20]
    assert landmarks == {0, 10, 17, 18, 19}
    assert int(frames[-1][1].mean()) == pytest.approx(200, abs=2)
