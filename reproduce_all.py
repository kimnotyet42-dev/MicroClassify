"""Run all manuscript experiments from the repository manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


CODE_ROOT = Path(__file__).resolve().parent
MANIFEST = CODE_ROOT / "experiment_manifest.json"


def load_manifest() -> Dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def run_module(module: str, arguments: List[str]) -> None:
    command = [sys.executable, "-m", module] + arguments
    print("Running: {0}".format(" ".join(command)), flush=True)
    subprocess.check_call(command, cwd=str(CODE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    selected = set(args.only or [])
    known = {
        item["id"] for item in manifest["experiments"] + manifest["plots"]
    }
    unknown = selected - known
    if unknown:
        parser.error("Unknown experiment IDs: {0}".format(sorted(unknown)))

    for item in manifest["experiments"]:
        if selected and item["id"] not in selected:
            continue
        if args.skip_runtime and item["id"] == "runtime":
            continue
        command_args = list(item.get("args", []))
        if item.get("resume") and not args.no_resume:
            command_args.append("--resume")
        if args.config is not None:
            command_args.extend(["--config", str(args.config)])
        run_module(item["module"], command_args)

    if not args.skip_plots:
        for item in manifest["plots"]:
            if selected and item["id"] not in selected:
                continue
            command_args = []
            if args.config is not None:
                command_args.extend(["--config", str(args.config)])
            run_module(item["module"], command_args)


if __name__ == "__main__":
    main()
