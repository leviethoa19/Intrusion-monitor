import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "detect_people.py"


def load_module():
    spec = importlib.util.spec_from_file_location("detect_people", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalized_source_converts_webcam_index():
    module = load_module()
    assert module.normalized_source("0") == 0
    assert module.normalized_source("video.mp4") == "video.mp4"


def test_safe_name_removes_unsafe_characters():
    module = load_module()
    assert module.safe_name("data/raw/test video.mp4") == "test_video"

