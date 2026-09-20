"""Benchmark solver strategies on the real-game fixture plus random boards.

Examples::

    python benchmark.py                       # quick sweep, 200 random boards
    python benchmark.py --boards 20 --budget 2.0 --only rollout
    python benchmark.py --plan-sweep          # mean score vs planning budget
"""

from __future__ import annotations

import _env  # noqa: F401  checks the venv before the imports below

import argparse
import multiprocessing as mp
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import solver

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "board_real.txt"

PLAN_BUDGETS = (1.0, 3.0, 5.0, 10.0, 20.0, 40.0, 60.0)
EXECUTE_SECONDS_PER_MOVE = 0.42    # measured in the browser, see bot.DRAG_STEPS
TIME_LIMIT = 110.0                 # planning + execution must fit inside this


def build_strategies(
    budget: float, total: float | None = None
) -> dict[str, solver.Strategy]:
    """Strategy line-up.

    ``budget`` caps think time per move. ``total``, if given, is a whole-game
    think budget the rollout splits across its moves exactly as the live bot
    does - the honest way to compare against the 120 second clock.
    """
    return {
        "greedy-fewest": solver.GreedyFewest(),
        "greedy-most": solver.GreedyMost(),
        "random-uniform": solver.RandomPolicy(alpha=0.0),
        "random-biased": solver.RandomPolicy(alpha=12.0),
        "beam": solver.Beam(width=24, time_budget=budget),
        "rollout-plain": solver.Rollout(time_budget=budget, total_budget=total,
                                        alpha=2.0, keep_line=False, halving=False),
        "rollout": solver.Rollout(time_budget=budget, total_budget=total),
    }


@dataclass
class Run:
    score: int
    moves: int
    elapsed: float


def _play(args: tuple[str, float, float | None, np.ndarray, int]) -> Run:
    name, budget, total, board, seed = args
    strategy = build_strategies(budget, total)[name]
    result = solver.simulate(board, strategy, random.Random(seed))
    return Run(result.score, result.move_count, result.elapsed)


def summarise(name: str, runs: list[Run], fixture: Run) -> str:
    scores = [r.score for r in runs]
    per_move = [r.elapsed / r.moves for r in runs if r.moves]
    return (
        f"{name:<15} {statistics.mean(scores):7.1f} {statistics.median(scores):7.0f} "
        f"{min(scores):6d} {max(scores):6d} "
        f"{statistics.pstdev(scores):6.1f} "
        f"{statistics.mean(per_move) * 1000:9.1f} "
        f"{statistics.mean([r.moves for r in runs]):7.1f} "
        f"{statistics.mean([r.elapsed for r in runs]):8.2f} "
        f"{fixture.score:8d}"
    )


# --------------------------------------------------------------------------
# Planning budget sweep
# --------------------------------------------------------------------------

def _plan(args: tuple[float, np.ndarray, int]) -> tuple[int, int, float]:
    budget, board, seed = args
    result = solver.plan(board, budget, random.Random(seed))
    return result.score, len(result.moves), result.elapsed


def chart(budgets: Sequence[float], means: Sequence[float], height: int = 14) -> str:
    """A small ASCII plot of mean score against planning budget."""
    low, high = min(means), max(means)
    span = high - low or 1.0
    pad = span * 0.1
    low, high = low - pad, high + pad
    step = (high - low) / height
    columns = [f"{b:g}s" for b in budgets]
    width = max(len(c) for c in columns) + 3
    lines = []
    for row in range(height, -1, -1):
        level = low + row * step
        marks = "".join(
            ("*" if abs(m - level) <= step / 2 else
             ("|" if m > level else " ")).center(width)
            for m in means
        )
        label = f"{level:6.1f}" if row % 2 == 0 else " " * 6
        lines.append(f"{label} |{marks}")
    lines.append(" " * 6 + " +" + "-" * (width * len(means)))
    lines.append(" " * 6 + "  " + "".join(c.center(width) for c in columns))
    lines.append(" " * 6 + "  " + "planning budget".center(width * len(means)))
    return "\n".join(lines)


def plan_sweep(args: argparse.Namespace) -> None:
    """Score whole-game plans across a range of planning budgets."""
    rng = np.random.default_rng(args.seed)
    boards = [solver.random_board(rng) for _ in range(args.boards)]
    fixture = solver.parse_board(FIXTURE.read_text())

    print(f"{args.boards} random boards + fixture | jobs {args.jobs} | "
          f"max score {solver.ROWS * solver.COLS}")
    print(f"{'budget':>7} {'mean':>7} {'gain':>6} {'median':>7} {'max':>5} "
          f"{'moves':>6} {'exec':>6} {'total':>7} {'fixture':>8}")
    print("-" * 70)

    means: list[float] = []
    previous: float | None = None
    for budget in PLAN_BUDGETS:
        fixture_score, _, _ = _plan((budget, fixture, args.seed))
        jobs = [(budget, b, args.seed + i) for i, b in enumerate(boards)]
        if args.jobs > 1:
            with mp.Pool(args.jobs) as pool:
                runs = pool.map(_plan, jobs)
        else:
            runs = [_plan(j) for j in jobs]
        scores = [r[0] for r in runs]
        moves = statistics.mean(r[1] for r in runs)
        execute = moves * EXECUTE_SECONDS_PER_MOVE
        total = budget + execute
        mean = statistics.mean(scores)
        means.append(mean)
        gain = "-" if previous is None else f"{mean - previous:+.1f}"
        flag = "" if total <= TIME_LIMIT else "  OVER LIMIT"
        previous = mean
        print(f"{budget:6.0f}s {mean:7.1f} {gain:>6} {statistics.median(scores):7.0f} "
              f"{max(scores):5d} {moves:6.1f} {execute:5.0f}s {total:6.0f}s "
              f"{fixture_score:8d}{flag}", flush=True)

    print("-" * 70)
    print(f"exec = {EXECUTE_SECONDS_PER_MOVE}s per drag; "
          f"total must stay under {TIME_LIMIT:.0f}s\n")
    print(chart(PLAN_BUDGETS, means))
    if args.csv:
        rows = ["budget_s,mean_score"] + [f"{b:g},{m:.2f}"
                                          for b, m in zip(PLAN_BUDGETS, means)]
        Path(args.csv).write_text("\n".join(rows) + "\n")
        print(f"\nwrote {args.csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=200, help="random boards per strategy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=0.05,
                        help="cap on per-move think time (s) for rollout/beam")
    parser.add_argument("--total-budget", type=float, default=None,
                        help="whole-game think budget (s); ~60 matches the live bot. "
                             "Raises the per-move cap unless --budget is given too")
    parser.add_argument("--only", nargs="*", help="subset of strategy names")
    parser.add_argument("--jobs", type=int, default=mp.cpu_count() - 1)
    parser.add_argument("--plan-sweep", action="store_true",
                        help="mean score vs planning budget, for picking one")
    parser.add_argument("--csv", help="write the sweep results here")
    args = parser.parse_args()
    if args.plan_sweep:
        plan_sweep(args)
        return
    if args.total_budget and args.budget == parser.get_default("budget"):
        # Otherwise the per-move cap silently swallows the whole-game budget.
        args.budget = args.total_budget / 8

    names = list(build_strategies(args.budget, args.total_budget))
    if args.only:
        unknown = set(args.only) - set(names)
        if unknown:
            parser.error(f"unknown strategies: {sorted(unknown)} (have {names})")
        names = args.only

    rng = np.random.default_rng(args.seed)
    boards = [solver.random_board(rng) for _ in range(args.boards)]
    fixture = solver.parse_board(FIXTURE.read_text())

    how = (f"whole-game budget {args.total_budget}s" if args.total_budget
           else f"per-move budget {args.budget}s")
    print(f"{args.boards} random boards + fixture | {how} "
          f"| jobs {args.jobs} | max score {solver.ROWS * solver.COLS}")
    print(f"{'strategy':<15} {'mean':>7} {'median':>7} {'min':>6} {'max':>6} "
          f"{'stdev':>6} {'ms/move':>9} {'moves':>7} {'s/game':>8} {'fixture':>8}")
    print("-" * 95)

    started = time.perf_counter()
    for name in names:
        fixture_run = _play((name, args.budget, args.total_budget, fixture, args.seed))
        jobs = [(name, args.budget, args.total_budget, b, args.seed + i)
                for i, b in enumerate(boards)]
        if args.jobs > 1:
            with mp.Pool(args.jobs) as pool:
                runs = pool.map(_play, jobs)
        else:
            runs = [_play(j) for j in jobs]
        print(summarise(name, runs, fixture_run), flush=True)
    print("-" * 95)
    print(f"total {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
