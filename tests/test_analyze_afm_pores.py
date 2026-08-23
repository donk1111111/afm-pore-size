from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_afm_pores.py"


class AnalyzeAfmPoresTests(unittest.TestCase):
    def test_self_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--self-test", "--output", str(tmp_path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            detail_path = tmp_path / "pores_detail.csv"
            with detail_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(float(row["Feret"]) > 0 for row in rows))
            self.assertTrue(all(row["unit"] == "um" for row in rows))

    def test_manual_unit_per_px(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--self-test",
                    "--output",
                    str(tmp_path),
                    "--unit-per-px",
                    "0.05",
                    "--unit",
                    "um",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()