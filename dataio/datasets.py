from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    key: str
    path: Path
    label: str

    def open_image(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")


def load_emds7(directory: Path) -> List[Sample]:
    """Load EMDS-7 from the directory that directly contains its PNG files."""
    directory = Path(directory)
    samples = []
    for path in sorted(directory.glob("*.png")):
        match = re.match(r"EMDS7-(G\d+)-", path.name)
        if match:
            samples.append(Sample(path.name, path, match.group(1)))
    if not samples:
        raise FileNotFoundError(f"No EMDS-7 images found under {directory}")
    return samples


def _class_directory_root(directory: Path, extension: str) -> Path:
    class_directories = [
        path
        for path in directory.rglob("*")
        if path.is_dir() and any(path.glob(extension))
    ]
    if not class_directories:
        raise FileNotFoundError(f"No class directories matching {extension} under {directory}")
    return Counter(path.parent for path in class_directories).most_common(1)[0][0]


def load_zoolake2(directory: Path) -> List[Sample]:
    """Load ZooLake2 from its extracted dataset directory."""
    directory = Path(directory)
    root = _class_directory_root(directory, "*.jpeg")
    samples = []
    for class_directory in sorted(root.iterdir(), key=lambda path: path.name):
        if not class_directory.is_dir():
            continue
        for path in sorted(class_directory.glob("*.jpeg"), key=lambda item: item.name):
            samples.append(
                Sample(
                    f"{class_directory.name}/{path.name}",
                    path,
                    class_directory.name,
                )
            )
    if not samples:
        raise FileNotFoundError(f"No ZooLake2 images found under {directory}")
    return samples


def load_self_collected(microscopy_root: Path) -> List[Sample]:
    microscopy_root = Path(microscopy_root)
    samples = []
    for class_directory in sorted(microscopy_root.iterdir(), key=lambda path: path.name):
        if not class_directory.is_dir():
            continue
        for path in sorted(class_directory.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(
                    Sample(
                        f"{class_directory.name}/{path.name}",
                        path,
                        class_directory.name,
                    )
                )
    if not samples:
        raise FileNotFoundError(f"No microscopy images found under {microscopy_root}")
    return samples
