"""Play Fruit Box in a real browser.

Opens the game in headed Chromium, presses Play, then loops: screenshot the
canvas -> parse the board -> pick a move -> drag it -> wait for the clear
animation. Stops when no legal move remains or the 120 second clock runs out.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import Page, sync_playwright

import solver
import vision

URL = "https://en.gamesaien.com/game/fruit_box/"
GAME_SECONDS = 120.0
ROOT = Path(__file__).resolve().parent
DEBUG_DIR = ROOT / "debug"

# Leave room for the final drag plus a safety margin when budgeting think time.
END_RESERVE = 2.0
APPLES_PER_MOVE = 3.2      # rough average, used to estimate moves remaining


@dataclass
class Geometry:
    """Maps screenshot pixels to page (CSS) coordinates."""

    x: float
    y: float
    width: float
    height: float
    scale: float          # screenshot pixels per CSS pixel (2.0 on Retina)

    def to_page(self, px: float, py: float) -> tuple[float, float]:
        return self.x + px / self.scale, self.y + py / self.scale


class GameSession:
    """A browser tab with the game loaded."""

    def __init__(self, headless: bool = False, slow_mo: float = 0.0) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self._pw = None
        self._browser = None
        self.page: Page | None = None
        self.geometry: Geometry | None = None

    def __enter__(self) -> "GameSession":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo
        )
        context = self._browser.new_context(
            viewport={"width": 1400, "height": 1000}, device_scale_factor=2
        )
        self.page = context.new_page()
        # The game only paints once the window "load" event fires (it waits on
        # webfont.js and createjs), so an early screenshot is a blank canvas.
        self.page.goto(URL, wait_until="load")
        self.page.wait_for_selector("#canvas", timeout=30_000)
        self._measure()
        self.wait_for_title(timeout=45.0)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    # -- geometry ---------------------------------------------------------

    def _measure(self) -> None:
        assert self.page is not None
        box = self.page.locator("#canvas").bounding_box()
        if box is None:
            raise RuntimeError("canvas has no bounding box")
        shot = self.screenshot(measure=False)
        # Derive the scale from the screenshot itself rather than trusting the
        # device pixel ratio, so Retina and non-Retina both work.
        scale = shot.shape[1] / box["width"]
        self.geometry = Geometry(box["x"], box["y"], box["width"], box["height"], scale)

    def screenshot(self, measure: bool = True) -> np.ndarray:
        """Canvas-only screenshot as a BGR array."""
        assert self.page is not None
        raw = self.page.locator("#canvas").screenshot()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if measure and self.geometry is None:
            self._measure()
        return image

    def to_page(self, px: float, py: float) -> tuple[float, float]:
        assert self.geometry is not None
        return self.geometry.to_page(px, py)

    # -- game control -----------------------------------------------------

    def wait_for_title(self, timeout: float = 45.0) -> np.ndarray:
        """Poll until the title screen has actually been painted."""
        assert self.page is not None
        deadline = time.perf_counter() + timeout
        while True:
            image = self.screenshot()
            if float(vision.red_mask(image).mean()) > 0.01:
                return image
            if time.perf_counter() >= deadline:
                raise RuntimeError("the game never finished loading")
            self.page.wait_for_timeout(250)

    def press_play(self) -> None:
        """Click the big Play apple: the largest red blob on the title screen."""
        image = self.wait_for_title()
        mask = vision.red_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("no Play button found on the title screen")
        moments = cv2.moments(max(contours, key=cv2.contourArea))
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        assert self.page is not None
        self.page.mouse.click(*self.to_page(cx, cy))

    def wait_for_board(self, timeout: float = 15.0) -> tuple[np.ndarray, vision.Calibration]:
        """Poll until a full 17x10 grid of apples is on screen."""
        deadline = time.perf_counter() + timeout
        last: Exception | None = None
        while time.perf_counter() < deadline:
            image = self.screenshot()
            try:
                return image, vision.detect_grid(image)
            except vision.CalibrationError as exc:
                last = exc
            assert self.page is not None
            self.page.wait_for_timeout(250)
        raise RuntimeError(f"board never appeared: {last}")

    def drag(self, move: solver.Move, cal: vision.Calibration, steps: int = 8) -> None:
        """Drag a selection rectangle around ``move``.

        Start and end sit in the gaps between apples, half a cell outside the
        target rectangle, so the selection encloses exactly the wanted cells.
        """
        assert self.page is not None
        x0, y0 = cal.corner(move.r1, move.c1)
        x1, y1 = cal.corner(move.r2 + 1, move.c2 + 1)
        sx, sy = self.to_page(x0, y0)
        ex, ey = self.to_page(x1, y1)
        mouse = self.page.mouse
        mouse.move(sx, sy)
        mouse.down()
        for i in range(1, steps + 1):
            t = i / steps
            mouse.move(sx + (ex - sx) * t, sy + (ey - sy) * t)
        mouse.up()


# --------------------------------------------------------------------------
# Board reading with animation handling
# --------------------------------------------------------------------------

def wait_for_count(
    session: GameSession,
    cal: vision.Calibration,
    expected_apples: int,
    timeout: float = 1.2,
) -> tuple[int, np.ndarray]:
    """Poll until the apple count matches, i.e. the clear animation is done.

    Apples fade out rather than vanishing, so a screenshot taken immediately
    after the drag still shows them. Counting apples needs no digit matching,
    which makes this much cheaper than a full re-read.
    """
    deadline = time.perf_counter() + timeout
    while True:
        image = session.screenshot()
        count = int(vision.apple_grid(image, cal).sum())
        if count == expected_apples or time.perf_counter() >= deadline:
            return count, image
        assert session.page is not None
        session.page.wait_for_timeout(50)


def think_budget(deadline: float, board: solver.Board, floor: float = 0.02) -> float:
    """Split the remaining clock across the moves we still expect to make."""
    left = deadline - time.perf_counter() - END_RESERVE
    if left <= 0:
        return floor
    moves_left = max(1.0, solver.remaining_apples(board) / APPLES_PER_MOVE)
    return max(floor, left / moves_left)


def make_strategy(name: str, budget: float, alpha: float) -> solver.Strategy:
    if name == "rollout":
        return solver.Rollout(time_budget=budget, alpha=alpha)
    if name == "greedy":
        return solver.GreedyFewest()
    if name == "beam":
        return solver.Beam(time_budget=budget)
    raise ValueError(f"unknown strategy {name!r}")


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def run_dry(args: argparse.Namespace) -> None:
    """Read the board, print it and the planned line, drag nothing."""
    reader = vision.DigitReader.load()
    with GameSession(headless=args.headless) as session:
        session.press_play()
        image, detected = session.wait_for_board()
        cal = load_calibration(detected, args.recalibrate)
        reading = vision.read_board(image, cal, reader)
        print(solver.board_to_str(reading.board))
        print(f"\napples read: {reading.apples}/170  "
              f"weakest match: {reading.weakest:.2f}")
        weak = reading.low_confidence_cells()
        if weak:
            print(f"low-confidence cells: {weak}")

        strategy = make_strategy(args.strategy, args.budget, args.alpha)
        result = solver.simulate(reading.board, strategy, random.Random(args.seed))
        print(f"\nplanned line: {result.move_count} moves, "
              f"score {result.score}, {result.elapsed:.1f}s of thinking")
        for i, move in enumerate(result.moves, 1):
            x0, y0 = cal.corner(move.r1, move.c1)
            x1, y1 = cal.corner(move.r2 + 1, move.c2 + 1)
            print(f"{i:3d}. rows {move.r1}-{move.r2} cols {move.c1}-{move.c2} "
                  f"{move.apples} apples   drag {session.to_page(x0, y0)} -> "
                  f"{session.to_page(x1, y1)}")
        if args.debug:
            save_debug(image, reading, result.moves[0] if result.moves else None, 0)


def run_play(args: argparse.Namespace) -> None:
    reader = vision.DigitReader.load()
    rng = random.Random(args.seed)
    with GameSession(headless=args.headless) as session:
        session.press_play()
        started = time.perf_counter()
        image, detected = session.wait_for_board()
        cal = load_calibration(detected, args.recalibrate)
        deadline = started + GAME_SECONDS
        reading = vision.read_board(image, cal, reader)
        board = reading.board
        print(solver.board_to_str(board))
        if reading.apples != solver.ROWS * solver.COLS:
            print(f"warning: only read {reading.apples}/170 apples")

        score = 0
        played = 0
        while time.perf_counter() < deadline - END_RESERVE:
            budget = think_budget(deadline, board, args.floor)
            strategy = make_strategy(args.strategy, min(budget, args.budget), args.alpha)
            move = strategy.choose(board, rng)
            if move is None:
                print("no legal moves left")
                break
            session.drag(move, cal)
            played += 1
            score += move.apples
            expected = solver.remaining_apples(board) - move.apples
            board = solver.apply_move(board, move, inplace=True)

            count, image = wait_for_count(session, cal, expected)
            if count != expected:
                print(f"  move {played}: expected {expected} apples on screen, "
                      f"saw {count} - forcing a full re-read")
            # A full re-read costs a round of template matching, so do it on the
            # configured interval, and always when the count looks wrong.
            if count != expected or played % args.reread_every == 0:
                reading = vision.read_board(image, cal, reader)
                if reading.weakest >= vision.MIN_MATCH_SCORE:
                    board = reading.board          # trust the screen over our model
                    score = solver.ROWS * solver.COLS - reading.apples
                else:
                    print(f"  move {played}: weak digit match "
                          f"{reading.weakest:.2f} at {reading.low_confidence_cells()}"
                          f" - keeping the tracked board")
                if args.debug:
                    save_debug(image, reading, move, played)
            print(f"{played:3d}. rows {move.r1}-{move.r2} cols {move.c1}-{move.c2} "
                  f"+{move.apples} = {score}  ({budget * 1000:.0f}ms think, "
                  f"{deadline - time.perf_counter():.0f}s left)")

        image = session.screenshot()
        final = vision.read_board(image, cal, reader)
        cleared = solver.ROWS * solver.COLS - final.apples
        print(f"\nfinished: {played} moves, {cleared} apples cleared "
              f"(tracked score {score}) in {time.perf_counter() - started:.1f}s")
        if args.debug:
            save_debug(image, final, None, 999)
        if args.hold:
            print(f"holding the window open for {args.hold}s")
            assert session.page is not None
            session.page.wait_for_timeout(int(args.hold * 1000))


def load_calibration(detected: vision.Calibration, recalibrate: bool) -> vision.Calibration:
    """Prefer the saved calibration; fall back to what we just detected."""
    if not recalibrate and vision.CALIBRATION_PATH.exists():
        saved = vision.Calibration.load()
        return saved.rescaled(detected.image_w, detected.image_h)
    return detected


def save_debug(
    image: np.ndarray,
    reading: vision.BoardReading,
    move: solver.Move | None,
    index: int,
) -> None:
    DEBUG_DIR.mkdir(exist_ok=True)
    caption = f"move {index} | {reading.apples} apples | min match {reading.weakest:.2f}"
    cv2.imwrite(str(DEBUG_DIR / f"board_{index:03d}.png"),
                vision.annotate(image, reading, move, caption))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="read the board and print the plan without dragging")
    parser.add_argument("--debug", action="store_true",
                        help="save annotated screenshots to debug/")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--strategy", default="rollout",
                        choices=["rollout", "greedy", "beam"])
    parser.add_argument("--budget", type=float, default=2.0,
                        help="upper bound on per-move think time (s)")
    parser.add_argument("--floor", type=float, default=0.02,
                        help="lower bound on per-move think time (s)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="rollout policy bias towards small clears")
    parser.add_argument("--reread-every", type=int, default=1,
                        help="re-read the board every N moves")
    parser.add_argument("--recalibrate", action="store_true",
                        help="ignore calibration.json and detect the grid live")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold", type=float, default=0.0,
                        help="keep the browser open this long after the game")
    args = parser.parse_args()
    (run_dry if args.dry_run else run_play)(args)


if __name__ == "__main__":
    main()
