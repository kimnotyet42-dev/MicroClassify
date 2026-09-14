from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime_config import CODE_ROOT, load_paths


class RuntimeConfigTests(unittest.TestCase):
    def test_ini_paths_are_loaded_from_one_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "emds7"
            model = root / "dinov3"
            dataset.mkdir()
            model.mkdir()
            config = root / "paths.ini"
            config.write_text(
                "[datasets]\n"
                "emds7 = {0}\n"
                "[models]\n"
                "dinov3 = {1}\n"
                "[outputs]\n"
                "cache = cache\n"
                "results = results\n"
                "figures = figures\n".format(dataset.as_posix(), model.as_posix()),
                encoding="utf-8",
            )
            paths = load_paths(config)
            self.assertEqual(paths.dataset("emds7"), dataset)
            self.assertEqual(paths.model("dinov3"), model)
            self.assertEqual(paths.output("cache"), (CODE_ROOT / "cache").resolve())


if __name__ == "__main__":
    unittest.main()
