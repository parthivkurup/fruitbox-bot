import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import solver  # noqa: E402

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "board_real.txt"


@pytest.fixture
def real_board() -> solver.Board:
    """Board transcribed from a real game."""
    return solver.parse_board(FIXTURE_PATH.read_text())
