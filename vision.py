"""Read a Fruit Box board from a screenshot of the game canvas.

The canvas is the only source of truth (the DOM holds no board data), so the
pipeline is: mask the red apples -> locate the grid -> cut each cell -> lift the
white digit out of the apple silhouette -> template match it against
``templates/<digit>.png``.

All pixel coordinates in this module are *screenshot* pixels, i.e. the canvas
backing store, which on a Retina display is 2x the CSS size. ``bot.py`` converts
back to CSS coordinates using the measured screenshot scale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import solver

ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "calibration.json"
TEMPLATES_DIR = ROOT / "templates"

GLYPH_SIZE = (24, 32)          # (width, height) every glyph is normalised to
DIGITS = tuple(range(1, 10))

# An apple covers roughly half of its cell crop; a cleared cell has no red at
# all. The midpoint is a wide, safe margin.
EMPTY_RED_FRACTION = 0.15
MIN_GLYPH_PIXELS = 20
MIN_MATCH_SCORE = 0.45


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Where the grid sits in the canvas, in screenshot pixels."""

    origin_x: float          # centre of column 0
    origin_y: float          # centre of row 0
    pitch_x: float
    pitch_y: float
    image_w: int             # screenshot size this was measured at
    image_h: int

    def centre(self, row: int, col: int) -> tuple[float, float]:
        return (self.origin_x + col * self.pitch_x,
                self.origin_y + row * self.pitch_y)

    @property
    def half(self) -> int:
        """Half-width of a cell crop, a touch under half the pitch."""
        return int(min(self.pitch_x, self.pitch_y) * 0.5) - 1

    def corner(self, row: int, col: int) -> tuple[float, float]:
        """Top-left of the cell's *gap* box, halfway to the previous cell."""
        x, y = self.centre(row, col)
        return x - self.pitch_x / 2, y - self.pitch_y / 2

    def rescaled(self, image_w: int, image_h: int) -> "Calibration":
        """Same grid measured on a differently sized screenshot."""
        if (image_w, image_h) == (self.image_w, self.image_h):
            return self
        fx, fy = image_w / self.image_w, image_h / self.image_h
        return Calibration(self.origin_x * fx, self.origin_y * fy,
                           self.pitch_x * fx, self.pitch_y * fy, image_w, image_h)

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Calibration":
        return cls(**json.loads(path.read_text()))


def red_mask(image: np.ndarray) -> np.ndarray:
    """1 where the pixel belongs to an apple (red, saturated)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (((hue < 12) | (hue > 168)) & (sat > 100) & (val > 100)).astype(np.uint8)


def _bands(projection: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Contiguous runs where the projection exceeds ``threshold``."""
    on = projection > threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(on) - 1))
    return runs


def _fit_axis(centres: list[float], expected: int) -> tuple[float, float]:
    """Least-squares origin and pitch through evenly spaced centres."""
    if len(centres) != expected:
        raise CalibrationError(
            f"found {len(centres)} apple bands, expected {expected}"
        )
    index = np.arange(expected, dtype=np.float64)
    pitch, origin = np.polyfit(index, np.asarray(centres, dtype=np.float64), 1)
    return float(origin), float(pitch)


class CalibrationError(RuntimeError):
    """The grid could not be located in the screenshot."""


def detect_grid(image: np.ndarray, *, fraction: float = 0.15) -> Calibration:
    """Locate the grid on a *full* board by projecting the red mask.

    Only reliable while every cell still holds an apple, i.e. at game start.
    """
    mask = red_mask(image)
    if mask.mean() < 0.05:
        raise CalibrationError("almost no apples in this screenshot")
    cols = mask.sum(axis=0)
    rows = mask.sum(axis=1)
    col_bands = _bands(cols, cols.max() * fraction)
    row_bands = _bands(rows, rows.max() * fraction)
    ox, px = _fit_axis([(a + b) / 2 for a, b in col_bands], solver.COLS)
    oy, py = _fit_axis([(a + b) / 2 for a, b in row_bands], solver.ROWS)
    return Calibration(ox, oy, px, py, image.shape[1], image.shape[0])


# --------------------------------------------------------------------------
# Glyph extraction
# --------------------------------------------------------------------------

def cell_glyph(
    mask: np.ndarray, cal: Calibration, row: int, col: int
) -> tuple[np.ndarray | None, float]:
    """Normalised digit glyph for one cell, plus that cell's red fraction.

    Returns ``(None, fraction)`` when the cell holds no apple. The digit is the
    hole in the apple: fill the apple's outline, then subtract the red pixels.
    """
    cx, cy = cal.centre(row, col)
    half = cal.half
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    if x0 < 0 or y0 < 0 or x1 > mask.shape[1] or y1 > mask.shape[0]:
        return None, 0.0
    cell = mask[y0:y1, x0:x1]
    fraction = float(cell.mean())
    if fraction < EMPTY_RED_FRACTION:
        return None, fraction

    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, fraction
    silhouette = np.zeros_like(cell)
    cv2.drawContours(silhouette, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    digit = ((silhouette == 1) & (cell == 0)).astype(np.uint8) * 255
    ys, xs = np.nonzero(digit)
    if ys.size < MIN_GLYPH_PIXELS:
        return None, fraction
    cropped = digit[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return cv2.resize(cropped, GLYPH_SIZE, interpolation=cv2.INTER_AREA), fraction


def glyph_grid(
    image: np.ndarray, cal: Calibration
) -> tuple[list[list[np.ndarray | None]], np.ndarray]:
    """Every cell's glyph plus the matching red-fraction grid."""
    mask = red_mask(image)
    cal = cal.rescaled(image.shape[1], image.shape[0])
    glyphs: list[list[np.ndarray | None]] = []
    fractions = np.zeros((solver.ROWS, solver.COLS), dtype=np.float32)
    for r in range(solver.ROWS):
        row: list[np.ndarray | None] = []
        for c in range(solver.COLS):
            glyph, fraction = cell_glyph(mask, cal, r, c)
            fractions[r, c] = fraction
            row.append(glyph)
        glyphs.append(row)
    return glyphs, fractions


# --------------------------------------------------------------------------
# Digit recognition
# --------------------------------------------------------------------------

class DigitReader:
    """Template matcher over the nine saved digit templates."""

    def __init__(self, templates: dict[int, np.ndarray]) -> None:
        missing = sorted(set(DIGITS) - set(templates))
        if missing:
            raise FileNotFoundError(f"missing templates for digits {missing}")
        self.templates = {d: t.astype(np.uint8) for d, t in templates.items()}

    @classmethod
    def load(cls, directory: Path = TEMPLATES_DIR) -> "DigitReader":
        templates: dict[int, np.ndarray] = {}
        for digit in DIGITS:
            path = directory / f"{digit}.png"
            if path.exists():
                templates[digit] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if not templates:
            raise FileNotFoundError(
                f"no templates in {directory} - run `python calibrate.py` first"
            )
        return cls(templates)

    def match(self, glyph: np.ndarray) -> tuple[int, float]:
        """Best-matching digit and its normalised correlation score."""
        best_digit, best_score = 0, -1.0
        for digit, template in self.templates.items():
            score = float(
                cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0]
            )
            if score > best_score:
                best_digit, best_score = digit, score
        return best_digit, best_score


@dataclass
class BoardReading:
    board: solver.Board                 # 0 where the cell is empty
    scores: np.ndarray                  # match confidence per cell, 0 if empty
    red_fraction: np.ndarray
    calibration: Calibration

    @property
    def apples(self) -> int:
        return solver.remaining_apples(self.board)

    @property
    def weakest(self) -> float:
        occupied = self.scores[self.board > 0]
        return float(occupied.min()) if occupied.size else 1.0

    def low_confidence_cells(self, threshold: float = MIN_MATCH_SCORE) -> list[tuple[int, int]]:
        occupied = (self.board > 0) & (self.scores < threshold)
        return [(int(r), int(c)) for r, c in np.argwhere(occupied)]


def apple_grid(image: np.ndarray, cal: Calibration) -> np.ndarray:
    """Boolean occupancy grid, without running digit recognition.

    Much cheaper than :func:`read_board` and enough to tell whether a clear
    animation has finished.
    """
    mask = red_mask(image)
    cal = cal.rescaled(image.shape[1], image.shape[0])
    half = cal.half
    grid = np.zeros((solver.ROWS, solver.COLS), dtype=bool)
    for r in range(solver.ROWS):
        for c in range(solver.COLS):
            cx, cy = cal.centre(r, c)
            x0, y0 = int(round(cx - half)), int(round(cy - half))
            cell = mask[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
            grid[r, c] = cell.size > 0 and cell.mean() >= EMPTY_RED_FRACTION
    return grid


def read_board(
    image: np.ndarray, cal: Calibration, reader: DigitReader
) -> BoardReading:
    """Parse a canvas screenshot into a board."""
    glyphs, fractions = glyph_grid(image, cal)
    board = np.zeros((solver.ROWS, solver.COLS), dtype=np.int8)
    scores = np.zeros((solver.ROWS, solver.COLS), dtype=np.float32)
    for r in range(solver.ROWS):
        for c in range(solver.COLS):
            glyph = glyphs[r][c]
            if glyph is None:
                continue
            digit, score = reader.match(glyph)
            board[r, c] = digit
            scores[r, c] = score
    return BoardReading(board, scores, fractions,
                        cal.rescaled(image.shape[1], image.shape[0]))


# --------------------------------------------------------------------------
# Debug output
# --------------------------------------------------------------------------

def annotate(
    image: np.ndarray,
    reading: BoardReading,
    move: solver.Move | None = None,
    caption: str = "",
) -> np.ndarray:
    """Draw the parsed digits (and optionally the planned drag) on the canvas."""
    out = image.copy()
    cal = reading.calibration
    for r in range(solver.ROWS):
        for c in range(solver.COLS):
            x, y = cal.centre(r, c)
            value = int(reading.board[r, c])
            weak = value and reading.scores[r, c] < MIN_MATCH_SCORE
            colour = (0, 0, 255) if weak else ((160, 160, 160) if not value else (0, 0, 0))
            cv2.putText(out, str(value), (int(x) - 22, int(y) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
    if move is not None:
        x0, y0 = cal.corner(move.r1, move.c1)
        x1, y1 = cal.corner(move.r2 + 1, move.c2 + 1)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 3)
    if caption:
        cv2.putText(out, caption, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 255), 2, cv2.LINE_AA)
    return out
