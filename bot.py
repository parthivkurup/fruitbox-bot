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
# Each mouse event costs a ~17ms round trip and the game samples the pointer
# once per frame, so drags are a trade-off: too few steps and the game drops the
# drag entirely. The first attempt is cheap, a retry is thorough.
DRAG_STEPS = 12
RETRY_DRAG_STEPS = 28
CLEAR_TIMEOUT = 1.2        # hard cap on waiting for a clear animation
DROP_AFTER = 0.40          # board untouched for this long means the drag was dropped
# A game runs to roughly this many moves before the board dries up, typically
# leaving 40-50 apples stranded. Budgeting off apples-remaining alone reserves
# time for clears that will never happen, so take whichever estimate is smaller.
EXPECTED_MOVES = 62
APPLES_PER_MOVE = 2.1


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

    def screenshot(self, measure: bool = True, lossless: bool = False) -> np.ndarray:
        """Canvas-only screenshot as a BGR array.

        JPEG is ~3x faster to capture than PNG and the artwork is flat colour,
        so it costs nothing in accuracy; calibration still uses PNG.
        """
        assert self.page is not None
        if self.geometry is None or lossless:
            raw = self.page.locator("#canvas").screenshot()
        else:
            g = self.geometry
            raw = self.page.screenshot(
                clip={"x": g.x, "y": g.y, "width": g.width, "height": g.height},
                type="jpeg", quality=90,
            )
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

    def drag(self, move: solver.Move, cal: vision.Calibration, steps: int = DRAG_STEPS) -> None:
        """Drag a selection rectangle around ``move``.

        Start and end sit in the gaps between apples, half a cell outside the
        target rectangle, so the selection encloses exactly the wanted cells.

        The game samples the pointer once per animation frame, so a drag made of
        a few big jumps is often dropped entirely - long thin rectangles were the
        worst affected. Plenty of small steps makes drags land reliably.
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
    unchanged_apples: int | None = None,
    timeout: float = CLEAR_TIMEOUT,
) -> tuple[int, np.ndarray]:
    """Poll until the apple count matches, i.e. the clear animation is done.

    Apples fade out rather than vanishing, so a screenshot taken immediately
    after the drag still shows them. Counting apples needs no digit matching,
    which makes this much cheaper than a full re-read.

    A clear always resolves within ~300ms, so if ``unchanged_apples`` is still
    on screen after :data:`DROP_AFTER` the game never saw the drag and there is
    no point waiting out the full timeout.
    """
    started = time.perf_counter()
    while True:
        image = session.screenshot()
        count = int(vision.apple_grid(image, cal).sum())
        if count == expected_apples:
            return count, image
        elapsed = time.perf_counter() - started
        if count == unchanged_apples and elapsed >= DROP_AFTER:
            return count, image
        if elapsed >= timeout:
            return count, image
        assert session.page is not None
        session.page.wait_for_timeout(30)


def play_move(
    session: GameSession,
    cal: vision.Calibration,
    move: solver.Move,
    expected_apples: int,
    attempts: int = 3,
) -> tuple[bool, np.ndarray]:
    """Drag ``move`` and confirm it landed, retrying a drag the game ignored.

    A dropped drag leaves the board completely untouched, which is safe to
    repeat. Anything else means our model of the board is wrong, so we give up
    and let the caller re-read the screen.
    """
    unchanged = expected_apples + move.apples
    for attempt in range(attempts):
        session.drag(move, cal, steps=DRAG_STEPS if attempt == 0 else RETRY_DRAG_STEPS)
        count, image = wait_for_count(session, cal, expected_apples, unchanged)
        if count == expected_apples:
            return True, image
        if count != unchanged:
            return False, image
    return False, image


def moves_remaining(board: solver.Board, played: int) -> float:
    """How many more moves this game is likely to get."""
    by_apples = solver.remaining_apples(board) / APPLES_PER_MOVE
    return max(3.0, min(float(EXPECTED_MOVES - played), by_apples))


def think_budget(
    deadline: float,
    board: solver.Board,
    played: int,
    overhead: float,
    floor: float = 0.02,
) -> float:
    """Split the remaining clock across the moves we still expect to make.

    Dragging and re-reading cost roughly ``overhead`` seconds per move, so only
    what is left after paying that is available for thinking.
    """
    left = deadline - time.perf_counter() - END_RESERVE
    left_over = moves_remaining(board, played)
    return max(floor, (left - left_over * overhead) / left_over)


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
        overhead = args.overhead          # seconds of drag + re-read per move
        while time.perf_counter() < deadline - END_RESERVE:
            budget = think_budget(deadline, board, played, overhead, args.floor)
            strategy = make_strategy(args.strategy, min(budget, args.budget), args.alpha)
            move = strategy.choose(board, rng)
            if move is None:
                print("no legal moves left")
                break
            expected = solver.remaining_apples(board) - move.apples
            acting = time.perf_counter()
            landed, image = play_move(session, cal, move, expected)
            played += 1
            if landed:
                score += move.apples
                board = solver.apply_move(board, move, inplace=True)
            else:
                print(f"  move {played}: drag did not land - re-reading the board")
            # A full re-read costs a round of template matching, so do it on the
            # configured interval, and always when a drag misbehaved.
            if not landed or played % args.reread_every == 0:
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
            # Smooth the measured overhead so the budget tracks reality.
            overhead = 0.7 * overhead + 0.3 * (time.perf_counter() - acting)
            gain = f"+{move.apples}" if landed else "  x"
            print(f"{played:3d}. rows {move.r1}-{move.r2} cols {move.c1}-{move.c2} "
                  f"{gain} = {score}  ({budget * 1000:.0f}ms think, "
                  f"{overhead * 1000:.0f}ms overhead, "
                  f"{deadline - time.perf_counter():.0f}s left)")

        assert session.page is not None
        session.page.wait_for_timeout(500)      # let the last clear finish
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
    parser.add_argument("--overhead", type=float, default=0.45,
                        help="initial estimate of drag + re-read seconds per move")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold", type=float, default=0.0,
                        help="keep the browser open this long after the game")
    args = parser.parse_args()
    (run_dry if args.dry_run else run_play)(args)


if __name__ == "__main__":
    main()
