"""Play Fruit Box in a real browser.

Plan then execute: open the game in headed Chromium, press Play, read the board
once, search for a whole line of play, then drag that line out as fast as the
game will take it. Every few moves the screen is checked against the expected
position, and any disagreement triggers a quick replan from what is actually
there.
"""

from __future__ import annotations

import _env  # noqa: F401  checks the venv before the imports below

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

# Each mouse event costs a ~17ms round trip and the game samples the pointer
# once per frame, so a drag made of too few steps is dropped outright. Measured
# over full sequences, 24 steps executes 91-100% of a plan; fewer steps is
# faster per drag but loses more moves, and a lost move derails everything
# planned after it.
DRAG_STEPS = 24
EXECUTE_SECONDS_PER_MOVE = 0.42    # measured cost of one drag at DRAG_STEPS
SETTLE_TIMEOUT = 2.0       # hard cap on waiting for the screen to stop moving
SETTLE_GAP_MS = 50         # spacing between the frames compared for stability
SETTLE_STABLE = 0.25       # the board must hold still this long to count as settled
BADGE_TIMEOUT = 1.5        # hard cap on waiting for a score badge to fly away
MOVE_ATTEMPTS = 3          # drags per move before giving up on it


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

def read_settled(
    session: GameSession,
    cal: vision.Calibration,
    reader: vision.DigitReader,
    timeout: float = SETTLE_TIMEOUT,
) -> tuple[vision.BoardReading, np.ndarray]:
    """Read the board, but only once the screen has stopped changing.

    A clear animation can still be in flight several hundred milliseconds after
    the drag, and a frame caught mid-animation misreads cells - which poisons
    the tracked board and triggers replans that were never needed.

    Two frames a few tens of milliseconds apart are *not* enough to prove the
    board has stopped: apples cross the "still there" threshold at a handful of
    discrete moments, so mid-animation frames often match by chance. The board
    has to hold still for :data:`SETTLE_STABLE` before it is believed.
    """
    deadline = time.perf_counter() + timeout
    previous: np.ndarray | None = None
    steady_since: float | None = None
    while True:
        image = session.screenshot()
        now = time.perf_counter()
        if vision.popup_present(image) and now < deadline:
            # A score badge is flying over the board; anything under it misreads.
            previous, steady_since = None, None
            assert session.page is not None
            session.page.wait_for_timeout(SETTLE_GAP_MS)
            continue
        grid = vision.apple_grid(image, cal)
        if previous is not None and np.array_equal(grid, previous):
            steady_since = steady_since or now
            if now - steady_since >= SETTLE_STABLE:
                return vision.read_board(image, cal, reader), image
        else:
            steady_since = None
        if now >= deadline:
            return vision.read_board(image, cal, reader), image
        previous = grid
        assert session.page is not None
        session.page.wait_for_timeout(SETTLE_GAP_MS)


def wait_for_badge(session: GameSession, timeout: float = BADGE_TIMEOUT) -> None:
    """Hold off until no score badge is in flight.

    Badges are drawn over the board, and a drag that starts under one is
    swallowed - the single biggest cause of a plan executing short. In the
    normal path :func:`read_settled` has already waited the badge out before
    the next drag, so this is only needed on a retry.
    """
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if not vision.popup_present(session.screenshot()):
            return
        assert session.page is not None
        session.page.wait_for_timeout(SETTLE_GAP_MS)


def execute_move(
    session: GameSession,
    cal: vision.Calibration,
    reader: vision.DigitReader,
    move: solver.Move,
    expected: solver.Board,
) -> tuple[bool, int, vision.BoardReading]:
    """Drag ``move`` and confirm the board matches ``expected`` afterwards.

    Drags are not perfectly reliable, and one silently dropped drag invalidates
    every later move in the plan, so each one is checked and retried. Returns
    (landed, attempts, the reading it settled on).
    """
    for attempt in range(1, MOVE_ATTEMPTS + 1):
        if attempt > 1:
            wait_for_badge(session)      # only needed when retrying
        session.drag(move, cal)
        reading, _ = read_settled(session, cal, reader)
        if np.array_equal(reading.board, expected):
            return True, attempt, reading
        if reading.apples != solver.remaining_apples(expected) + move.apples:
            # Something changed, but not what we asked for: repeating the drag
            # would be guesswork, so let the caller re-read and replan.
            return False, attempt, reading
    return False, MOVE_ATTEMPTS, reading


def make_plan(
    board: solver.Board, budget: float, args: argparse.Namespace,
    rng: random.Random,
) -> solver.Plan:
    """Search ``board`` for ``budget`` seconds and return a full line of play."""
    if args.strategy == "greedy":
        result = solver.simulate(board, solver.GreedyFewest(), rng)
        return solver.Plan(result.moves, result.score, result.elapsed)
    return solver.plan(board, budget, rng, alpha=args.alpha)


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

        plan = make_plan(reading.board, args.plan_budget, args, random.Random(args.seed))
        estimate = plan.elapsed + len(plan) * EXECUTE_SECONDS_PER_MOVE
        print(f"\nplanned line: {len(plan)} moves, score {plan.score}, "
              f"{plan.elapsed:.1f}s planning + about "
              f"{len(plan) * EXECUTE_SECONDS_PER_MOVE:.0f}s of dragging "
              f"= {estimate:.0f}s of the {args.time_limit:.0f}s limit")
        for i, move in enumerate(plan.moves, 1):
            x0, y0 = cal.corner(move.r1, move.c1)
            x1, y1 = cal.corner(move.r2 + 1, move.c2 + 1)
            print(f"{i:3d}. rows {move.r1}-{move.r2} cols {move.c1}-{move.c2} "
                  f"{move.apples} apples   drag {session.to_page(x0, y0)} -> "
                  f"{session.to_page(x1, y1)}")
        if args.debug:
            save_debug(image, reading, plan.moves[0] if plan.moves else None, 0)
        hold_window(session, args)


def affordable_moves(
    moves: list[solver.Move], deadline: float, move_cost: float
) -> int:
    """How many of ``moves`` the clock can actually execute."""
    left = deadline - time.perf_counter()
    return max(0, min(len(moves), int(left / move_cost)))


def describe_diff(expected: solver.Board, observed: solver.Board) -> list[tuple]:
    """Cells where the two boards disagree, as (row, col, expected, observed)."""
    return [(int(r), int(c), int(expected[r, c]), int(observed[r, c]))
            for r, c in np.argwhere(expected != observed)]


def classify(move: solver.Move, diff: list[tuple]) -> str:
    """Best guess at why the board is not where the plan says it should be."""
    if not diff:
        return "none"
    def in_move(r: int, c: int) -> bool:
        return move.r1 <= r <= move.r2 and move.c1 <= c <= move.c2

    # A cell that still holds an apple but a *different* digit was never about
    # this drag at all - the initial parse was wrong. Check that first, because
    # such a cell can sit anywhere on the board.
    if any(e != 0 and o != 0 and e != o for _, _, e, o in diff):
        return "vision misread (digit differs)"
    if any(not in_move(r, c) for r, c, _, _ in diff):
        # Cells beyond the drag rectangle changed: the box took in a neighbour.
        return "drag geometry (changed outside the rectangle)"
    if all(e == 0 and o != 0 for _, _, e, o in diff):
        # The target apples are still on screen: the game never saw the drag.
        return "dropped drag (nothing cleared)"
    return "tracking bug (unexplained)"


def run_play(args: argparse.Namespace) -> None:
    reader = vision.DigitReader.load()
    score_reader: vision.ScoreReader | None
    try:
        score_reader = vision.ScoreReader.load()
    except FileNotFoundError:
        score_reader = None
        print("note: no score templates, cannot read the on-screen counter")
    rng = random.Random(args.seed)

    with GameSession(headless=args.headless) as session:
        session.press_play()
        started = time.perf_counter()
        image, detected = session.wait_for_board()
        cal = load_calibration(detected, args.recalibrate)
        deadline = started + args.time_limit

        reading = vision.read_board(image, cal, reader)
        board = reading.board
        print(solver.board_to_str(board))
        if reading.apples != solver.ROWS * solver.COLS:
            print(f"warning: only read {reading.apples}/170 apples")

        budget = min(args.plan_budget, deadline - time.perf_counter())
        plan = make_plan(board, budget, args, rng)
        print(f"\nplanned {len(plan)} moves scoring {plan.score} "
              f"in {plan.elapsed:.1f}s")

        # Promising the whole plan is dishonest if the clock cannot execute it,
        # so the prediction covers only the prefix that fits. The per-move cost
        # is re-measured as we go.
        move_cost = args.move_cost
        queue = list(plan.moves)
        affordable = affordable_moves(plan.moves, deadline, move_cost)
        if affordable < len(plan.moves):
            print(f"  only {affordable} of {len(plan)} moves fit in the "
                  f"remaining {deadline - time.perf_counter():.0f}s at "
                  f"{move_cost:.2f}s per move")
            queue = queue[:affordable]
        predicted = sum(m.apples for m in queue)
        executed = replans = retries = 0
        first_divergence: tuple | None = None
        since_check = 0

        while time.perf_counter() < deadline:
            if not queue:
                reading, _ = read_settled(session, cal, reader)
                board = reading.board
                since_check = 0
                if not solver.generate_moves(board) or args.no_replan:
                    break
                left = deadline - time.perf_counter()
                if left <= args.replan_budget:
                    break
                plan = make_plan(board, min(args.replan_budget, left), args, rng)
                if not plan.moves:
                    break
                replans += 1
                predicted += plan.score
                print(f"     replan {replans}: {len(plan)} more moves "
                      f"for {plan.score} (prediction now {predicted})")
                queue = list(plan.moves)
                continue

            move = queue.pop(0)
            expected = solver.apply_move(board, move)
            acting = time.perf_counter()
            landed, attempts, reading = execute_move(
                session, cal, reader, move, expected)
            move_cost = 0.8 * move_cost + 0.2 * (time.perf_counter() - acting)
            retries += attempts - 1
            executed += 1
            since_check += 1

            if landed:
                board = expected
            else:
                diff = describe_diff(expected, reading.board)
                kind = classify(move, diff)
                if first_divergence is None:
                    first_divergence = (executed, move, diff, kind)
                    save_divergence(session, cal, reading, move, executed, diff, kind)
                print(f"     DIVERGENCE at move {executed} after {attempts} "
                      f"attempts: {kind}")
                print(f"       move rows {move.r1}-{move.r2} cols "
                      f"{move.c1}-{move.c2} ({move.apples} apples)")
                print(f"       cells (r,c,expected,observed): {diff[:8]}")
                board = reading.board
                queue = []
                continue

            print(f"{executed:3d}. rows {move.r1}-{move.r2} cols {move.c1}-{move.c2} "
                  f"+{move.apples}{'  (retried)' if attempts > 1 else ''}  "
                  f"({len(queue)} queued, {deadline - time.perf_counter():.0f}s left)")

            if since_check >= args.verify_every:
                since_check = 0
                if not np.array_equal(reading.board, board):
                    diff = describe_diff(board, reading.board)
                    print(f"     checkpoint at move {executed}: "
                          f"{len(diff)} cells differ: {diff[:8]}")

        final, image = read_settled(session, cal, reader)
        cleared = solver.ROWS * solver.COLS - final.apples
        on_screen = vision.read_score(image, score_reader) if score_reader else None
        elapsed = time.perf_counter() - started

        print(f"\n{'=' * 62}")
        print(f"predicted score : {predicted}")
        print(f"on-screen score : {on_screen if on_screen is not None else 'unreadable'}")
        print(f"apples cleared  : {cleared}")
        exact = on_screen == predicted
        print(f"result          : {'EXACT MATCH' if exact else 'SHORT by ' + str(predicted - (on_screen if on_screen is not None else 0))}")
        print(f"moves executed  : {executed} ({retries} drag retries, {replans} replans)")
        if first_divergence:
            index, move, diff, kind = first_divergence
            print(f"first divergence: move {index}, {kind}")
        else:
            print("first divergence: none")
        print(f"elapsed         : {elapsed:.1f}s of the {args.time_limit:.0f}s limit")
        print("=" * 62)

        if args.debug:
            save_debug(image, final, None, 999)
        if not args.headless:
            wait_out_clock(session, started + GAME_SECONDS)
            if args.debug:
                DEBUG_DIR.mkdir(exist_ok=True)
                cv2.imwrite(str(DEBUG_DIR / "score_screen.png"),
                            session.screenshot(lossless=True))
        hold_window(session, args)


def save_divergence(
    session: GameSession,
    cal: vision.Calibration,
    observed: vision.BoardReading,
    move: solver.Move,
    index: int,
    diff: list[tuple],
    kind: str,
) -> None:
    """Save an annotated picture of the board where the plan came apart."""
    DEBUG_DIR.mkdir(exist_ok=True)
    caption = f"move {index}: {kind} | {len(diff)} cells differ"
    image = session.screenshot(lossless=True)
    cv2.imwrite(str(DEBUG_DIR / f"divergence_{index:03d}.png"),
                vision.annotate(image, observed, move, caption))


def wait_out_clock(session: GameSession, deadline: float) -> None:
    """Let the game's own 120s clock expire so it shows its Score screen.

    The bot usually plays the board dry with time to spare, and the game only
    reveals the score once the clock runs out.
    """
    left = deadline - time.perf_counter()
    if left <= 0:
        return
    print(f"waiting {left:.0f}s for the game clock, so the score screen appears")
    assert session.page is not None
    session.page.wait_for_timeout(int(left * 1000) + 1500)


def hold_window(session: GameSession, args: argparse.Namespace) -> None:
    """Leave the finished game on screen instead of closing the browser."""
    if args.headless:
        return
    assert session.page is not None
    if args.hold:
        print(f"holding the window open for {args.hold:.0f}s")
        session.page.wait_for_timeout(int(args.hold * 1000))
        return
    try:
        input("\npress Enter to close the browser ")
    except (EOFError, KeyboardInterrupt):      # not attached to a terminal
        pass


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="read the board and print the plan without dragging")
    parser.add_argument("--debug", action="store_true",
                        help="save annotated screenshots to debug/")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--strategy", default="rollout",
                        choices=["rollout", "greedy"])
    parser.add_argument("--plan-budget", type=float, default=20.0,
                        help="seconds to spend planning the opening line")
    parser.add_argument("--replan-budget", type=float, default=3.0,
                        help="seconds to spend replanning after a desync")
    parser.add_argument("--move-cost", type=float, default=1.05,
                        help="initial estimate of seconds to execute one move")
    parser.add_argument("--no-replan", action="store_true",
                        help="stop when the plan ends instead of replanning")
    parser.add_argument("--verify-every", type=int, default=10,
                        help="check the screen against the plan every N moves")
    parser.add_argument("--time-limit", type=float, default=110.0,
                        help="cap on planning plus execution (the game allows 120)")
    parser.add_argument("--drag-steps", type=int, default=DRAG_STEPS,
                        help="mouse positions per drag; fewer is faster, less reliable")
    parser.add_argument("--alpha", type=float, default=12.0,
                        help="rollout policy bias towards small clears")
    parser.add_argument("--recalibrate", action="store_true",
                        help="ignore calibration.json and detect the grid live")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold", type=float, default=0.0,
                        help="close the browser after this many seconds instead of "
                             "waiting for you to press Enter")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    (run_dry if args.dry_run else run_play)(args)


if __name__ == "__main__":
    main()
