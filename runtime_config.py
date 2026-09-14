"""Load machine-specific paths from ``configs/paths.ini``.

The committed example file documents every supported path.  A reproducer copies
it to ``configs/paths.ini`` and edits only that local file; experiment protocols
remain fixed in source control.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = CODE_ROOT / "configs" / "paths.ini"


@dataclass(frozen=True)
class RuntimePaths:
    datasets: Dict[str, Path]
    models: Dict[str, Path]
    outputs: Dict[str, Path]
    config_path: Path

    def dataset(self, name: str) -> Path:
        return _required_existing(self.datasets, name, "dataset")

    def model(self, name: str) -> Path:
        return _required_existing(self.models, name, "model")

    def optional_model(self, name: str) -> Optional[Path]:
        path = self.models.get(name)
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(
                "Configured optional model path does not exist: {0}={1}".format(
                    name, path
                )
            )
        return path

    def output(self, name: str) -> Path:
        if name not in self.outputs:
            raise KeyError("Missing output path: {0}".format(name))
        return self.outputs[name]


def _required_existing(values: Dict[str, Path], name: str, kind: str) -> Path:
    if name not in values:
        raise KeyError("Missing {0} path: {1}".format(kind, name))
    path = values[name]
    if not path.exists():
        raise FileNotFoundError(
            "Configured {0} path does not exist: {1}={2}".format(kind, name, path)
        )
    return path


def _resolve(raw: str) -> Optional[Path]:
    value = raw.strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (CODE_ROOT / path).resolve()


def load_paths(config_path: Optional[Path] = None) -> RuntimePaths:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            "Path configuration not found: {0}. Copy configs/paths.example.ini "
            "to configs/paths.ini and edit the local paths.".format(path)
        )
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(str(path), encoding="utf-8")
    for section in ("datasets", "models", "outputs"):
        if section not in parser:
            raise KeyError("Missing [{0}] section in {1}".format(section, path))

    datasets = {}
    models = {}
    outputs = {}
    for key, value in parser["datasets"].items():
        resolved = _resolve(value)
        if resolved is not None:
            datasets[key] = resolved
    for key, value in parser["models"].items():
        resolved = _resolve(value)
        if resolved is not None:
            models[key] = resolved
    for key, value in parser["outputs"].items():
        resolved = _resolve(value)
        if resolved is not None:
            outputs[key] = resolved
    return RuntimePaths(datasets, models, outputs, path.resolve())
