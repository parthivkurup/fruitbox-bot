"""Regression tests for reading a board off a canvas screenshot."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import solver
import vision

FIXTURES = Path(__file__).parent / "fixtures"
CANVAS = FIXTURES / "canvas_full.png"
TRANSCRIPT = FIXTURES / "canvas_full.txt"


@pytest.fixture(scope="module")
def canvas() -> np.ndarray:
    image = cv2.imread(str(CANVAS))
    assert image is not None, f"missing fixture {CANVAS}"
    return image


@pytest.fixture(scope="module")
def reader() -> vision.DigitReader:
    return vision.DigitReader.load()


def test_detect_grid_finds_seventeen_by_ten(canvas: np.ndarray) -> None:
    cal = vision.detect_grid(canvas)
    assert cal.pitch_x == pytest.approx(66.0, abs=0.5)
    assert cal.pitch_y == pytest.approx(66.0, abs=0.5)
    # Every cell centre must land inside the image.
    for r, c in [(0, 0), (solver.ROWS - 1, solver.COLS - 1)]:
        x, y = cal.centre(r, c)
        assert 0 < x < canvas.shape[1] and 0 < y < canvas.shape[0]


def test_saved_calibration_agrees_with_live_detection(canvas: np.ndarray) -> None:
    saved = vision.Calibration.load()
    live = vision.detect_grid(canvas)
    assert saved.origin_x == pytest.approx(live.origin_x, abs=1.0)
    assert saved.origin_y == pytest.approx(live.origin_y, abs=1.0)
    assert saved.pitch_x == pytest.approx(live.pitch_x, abs=0.1)


def test_calibration_rescales_with_the_screenshot() -> None:
    cal = vision.Calibration(169.5, 179.0, 66.0, 66.0, 1440, 940)
    half = cal.rescaled(720, 470)
    assert (half.origin_x, half.pitch_x) == (84.75, 33.0)
    assert cal.rescaled(1440, 940) is cal


def test_read_board_matches_the_transcription(
    canvas: np.ndarray, reader: vision.DigitReader
) -> None:
    expected = solver.parse_board(TRANSCRIPT.read_text())
    reading = vision.read_board(canvas, vision.Calibration.load(), reader)
    assert reading.board.tolist() == expected.tolist()
    assert reading.apples == solver.ROWS * solver.COLS
    assert reading.weakest > vision.MIN_MATCH_SCORE
    assert reading.low_confidence_cells() == []


def test_cleared_cells_read_as_empty(
    canvas: np.ndarray, reader: vision.DigitReader
) -> None:
    """Paint the board background over some cells; they must read as 0."""
    cal = vision.Calibration.load()
    image = canvas.copy()
    top_left_y = int(cal.centre(0, 0)[1])
    background = image[top_left_y - 40, 100].tolist()   # board backdrop colour
    cleared = [(0, 0), (4, 8), (9, 16), (3, 3)]
    for r, c in cleared:
        x, y = cal.centre(r, c)
        cv2.rectangle(image, (int(x - 33), int(y - 33)), (int(x + 33), int(y + 33)),
                      background, -1)
    reading = vision.read_board(image, cal, reader)
    for r, c in cleared:
        assert reading.board[r, c] == 0, f"cell {(r, c)} should read as empty"
    assert reading.apples == solver.ROWS * solver.COLS - len(cleared)
    assert int(vision.apple_grid(image, cal).sum()) == reading.apples


def test_apple_grid_agrees_with_read_board(
    canvas: np.ndarray, reader: vision.DigitReader
) -> None:
    cal = vision.Calibration.load()
    reading = vision.read_board(canvas, cal, reader)
    assert np.array_equal(vision.apple_grid(canvas, cal), reading.board > 0)


def test_detect_grid_rejects_a_blank_canvas() -> None:
    blank = np.full((940, 1440, 3), 255, dtype=np.uint8)
    with pytest.raises(vision.CalibrationError):
        vision.detect_grid(blank)


def test_digit_reader_reports_missing_templates(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vision.DigitReader.load(tmp_path)


def test_annotate_draws_without_changing_the_original(
    canvas: np.ndarray, reader: vision.DigitReader
) -> None:
    cal = vision.Calibration.load()
    reading = vision.read_board(canvas, cal, reader)
    move = solver.generate_moves(reading.board)[0]
    before = canvas.copy()
    out = vision.annotate(canvas, reading, move, "test")
    assert np.array_equal(canvas, before)
    assert not np.array_equal(out, canvas)


# --------------------------------------------------------------------------
# The flying score badge
# --------------------------------------------------------------------------

BADGE = FIXTURES / "canvas_badge.jpg"


@pytest.fixture(scope="module")
def badge_frame() -> np.ndarray:
    image = cv2.imread(str(BADGE))
    assert image is not None, f"missing fixture {BADGE}"
    return image


def test_a_settled_board_has_no_yellow_at_all(canvas: np.ndarray) -> None:
    """The badge detector relies on yellow being unique to the badge."""
    assert int(vision.yellow_mask(canvas).sum()) == 0
    assert not vision.popup_present(canvas)


def test_badge_frame_is_detected(badge_frame: np.ndarray) -> None:
    assert int(vision.yellow_mask(badge_frame).sum()) > vision.POPUP_MIN_PIXELS
    assert vision.popup_present(badge_frame)


def test_badge_corrupts_the_cell_it_covers(
    badge_frame: np.ndarray, reader: vision.DigitReader
) -> None:
    """Why frames with a badge must never be parsed.

    The badge is flying over row 0 in this frame, and the cell underneath it
    reads as empty even though the apple is still there.
    """
    cal = vision.Calibration.load()
    reading = vision.read_board(badge_frame, cal, reader)
    covered = reading.board[0, 11]
    assert covered == 0, "fixture should show the badge blanking cell (0, 11)"


# --------------------------------------------------------------------------
# The on-screen score counter
# --------------------------------------------------------------------------

def test_reads_the_score_counter(badge_frame: np.ndarray, canvas: np.ndarray) -> None:
    score_reader = vision.ScoreReader.load()
    assert vision.read_score(canvas, score_reader) == 0        # fresh board
    assert vision.read_score(badge_frame, score_reader) == 2   # after one clear


def test_score_reader_needs_every_digit() -> None:
    score_reader = vision.ScoreReader.load()
    assert set(score_reader.templates) == set(vision.SCORE_DIGITS)


def test_score_reader_reports_missing_templates(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vision.ScoreReader.load(tmp_path)
