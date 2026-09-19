"""Calibrate the grid geometry and save one digit template per digit.

Grabs a screenshot of a fresh board, locates the grid, crops every cell, groups
the 170 glyphs into nine look-alike clusters, and asks you to name them. The
answers are written to ``templates/<digit>.png`` and the geometry to
``calibration.json``.

    python calibrate.py                      # open the game and calibrate
    python calibrate.py --image board.png    # calibrate from a saved screenshot
    python calibrate.py --labels 598671243   # skip the prompt
    python calibrate.py --save-cells cells/  # also dump every cell crop
"""

from __future__ import annotations

import _env  # noqa: F401  checks the venv before the imports below

import argparse
from pathlib import Path

import cv2
import numpy as np

import solver
import vision

CLUSTER_THRESHOLD = 0.75          # correlation above which two glyphs match
CLUSTER_SHEET = vision.TEMPLATES_DIR / "_clusters.png"


def _normalise(glyph: np.ndarray) -> np.ndarray:
    vector = glyph.astype(np.float32).ravel()
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def cluster_glyphs(
    glyphs: list[np.ndarray],
) -> tuple[list[np.ndarray], list[int]]:
    """Group look-alike glyphs; return one representative per group."""
    reps: list[np.ndarray] = []
    rep_glyphs: list[np.ndarray] = []
    labels: list[int] = []
    for glyph in glyphs:
        vector = _normalise(glyph)
        if reps:
            scores = np.array([float(vector @ r) for r in reps])
            best = int(scores.argmax())
            if scores[best] > CLUSTER_THRESHOLD:
                labels.append(best)
                continue
        labels.append(len(reps))
        reps.append(vector)
        rep_glyphs.append(glyph)
    return rep_glyphs, labels


def ascii_art(glyph: np.ndarray, width: int = 11, height: int = 7) -> list[str]:
    """A tiny terminal preview so you can label without opening a file."""
    small = cv2.resize(glyph, (width, height), interpolation=cv2.INTER_AREA)
    return ["".join("#" if v > 110 else "." for v in row) for row in small]


def show_clusters(rep_glyphs: list[np.ndarray], counts: list[int]) -> None:
    arts = [ascii_art(g) for g in rep_glyphs]
    header = "  ".join(f"{i + 1:^11d}" for i in range(len(arts)))
    print("\ncluster:")
    print(header)
    for line in range(len(arts[0])):
        print("  ".join(art[line] for art in arts))
    print("  ".join(f"{c:^11d}" for c in counts), "  <- cells in each cluster")


def save_sheet(rep_glyphs: list[np.ndarray], path: Path) -> None:
    sheet = np.hstack([np.pad(g, 6, constant_values=0) for g in rep_glyphs])
    sheet = cv2.resize(sheet, (sheet.shape[1] * 4, sheet.shape[0] * 4),
                       interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), 255 - sheet)


def grab_screenshot(headless: bool) -> np.ndarray:
    import bot                                   # imported lazily: needs playwright

    with bot.GameSession(headless=headless) as session:
        session.press_play()
        image, _ = session.wait_for_board()
        return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, help="calibrate from a saved screenshot")
    parser.add_argument("--labels", help="the nine cluster digits, left to right")
    parser.add_argument("--save-cells", type=Path, help="dump every cell crop here")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be saved, write nothing")
    args = parser.parse_args()

    image = cv2.imread(str(args.image)) if args.image else grab_screenshot(args.headless)
    if image is None:
        parser.error(f"could not read {args.image}")

    cal = vision.detect_grid(image)
    print(f"grid: origin ({cal.origin_x:.1f}, {cal.origin_y:.1f}) "
          f"pitch ({cal.pitch_x:.2f}, {cal.pitch_y:.2f}) "
          f"in a {cal.image_w}x{cal.image_h} screenshot")

    grid, fractions = vision.glyph_grid(image, cal)
    flat = [(r, c, g) for r, row in enumerate(grid) for c, g in enumerate(row)]
    found = [(r, c, g) for r, c, g in flat if g is not None]
    print(f"cells with an apple: {len(found)}/{solver.ROWS * solver.COLS} "
          f"(red fraction {fractions.min():.2f}-{fractions.max():.2f})")
    if len(found) != solver.ROWS * solver.COLS:
        print("warning: calibrate on a *fresh* board, every cell should be full")

    if args.save_cells:
        args.save_cells.mkdir(parents=True, exist_ok=True)
        for r, c, glyph in found:
            cv2.imwrite(str(args.save_cells / f"cell_{r}_{c}.png"), glyph)
        print(f"wrote {len(found)} cell crops to {args.save_cells}/")

    rep_glyphs, labels = cluster_glyphs([g for _, _, g in found])
    counts = [labels.count(i) for i in range(len(rep_glyphs))]
    print(f"grouped into {len(rep_glyphs)} distinct glyphs")
    if len(rep_glyphs) != len(vision.DIGITS):
        print("expected exactly 9 - the grid or the colour thresholds may be off")
        raise SystemExit(1)

    show_clusters(rep_glyphs, counts)
    save_sheet(rep_glyphs, CLUSTER_SHEET)
    print(f"\ncontact sheet: {CLUSTER_SHEET}")

    answer = args.labels or input(
        "type the nine digits above, left to right (e.g. 598671243): "
    )
    answer = answer.strip()
    if sorted(answer) != [str(d) for d in vision.DIGITS]:
        parser.error(f"expected a permutation of 1-9, got {answer!r}")

    if args.dry_run:
        print("dry run: nothing written")
        return

    vision.TEMPLATES_DIR.mkdir(exist_ok=True)
    for glyph, digit in zip(rep_glyphs, answer):
        cv2.imwrite(str(vision.TEMPLATES_DIR / f"{digit}.png"), glyph)
    cal.save()
    print(f"saved 9 templates to {vision.TEMPLATES_DIR}/ and {vision.CALIBRATION_PATH}")

    reading = vision.read_board(image, cal, vision.DigitReader.load())
    print("\nboard read back with the new templates:")
    print(solver.board_to_str(reading.board))
    print(f"\napples: {reading.apples}  weakest match: {reading.weakest:.3f}")
    weak = reading.low_confidence_cells()
    print(f"low-confidence cells: {weak}" if weak else "every cell matched confidently")


if __name__ == "__main__":
    main()
