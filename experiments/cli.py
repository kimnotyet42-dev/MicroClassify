"""Small shared helpers for reproducible experiment command lines."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from runtime_config import RuntimePaths, load_paths


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path configuration; defaults to configs/paths.ini",
    )


def runtime_paths(config: Optional[Path]) -> RuntimePaths:
    return load_paths(config)


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def output_path(
    requested: Optional[Path],
    paths: RuntimePaths,
    filename: str,
) -> Path:
    return Path(requested) if requested is not None else paths.output("results") / filename
