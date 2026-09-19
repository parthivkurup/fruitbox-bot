"""Benchmark solver strategies on the real-game fixture plus random boards.

Examples::

    python benchmark.py                       # quick sweep, 200 random boards
    python benchmark.py --boards 20 --budget 2.0 --only rollout
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

import numpy as np

import solver

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "board_real.txt"


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
    args = parser.parse_args()
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
