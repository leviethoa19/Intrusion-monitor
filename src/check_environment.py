"""Print the runtime information needed before running the detector."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Ultralytics: {package_version('ultralytics')}")
    print(f"OpenCV: {package_version('opencv-python')}")
    print(f"PyTorch: {package_version('torch')}")

    try:
        import torch
    except ImportError:
        print("CUDA available: unavailable because PyTorch is not installed")
        return

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {total_memory:.1f} GB")


if __name__ == "__main__":
    main()

