"""Make the source-layout package importable from repository scripts."""

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "g1_rickshaw_lab"


def add_project_source() -> None:
    sys.path.insert(0, str(SOURCE_ROOT))
