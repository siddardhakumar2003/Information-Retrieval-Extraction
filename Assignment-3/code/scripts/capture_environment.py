#!/usr/bin/env python3
"""Writes results/environment.txt: machine specs + library versions, per the
assignment's ground rule to note these for the report. Run as part of
run_all.sh so it always reflects whichever machine most recently ran the
benchmarks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annbench.envinfo import write_environment_info
from annbench.paths import RESULTS_DIR, ensure_output_dirs


def main() -> None:
    ensure_output_dirs()
    write_environment_info(RESULTS_DIR / "environment.txt")


if __name__ == "__main__":
    main()
