"""Tests for the browser-independent parts of the bot: geometry and budgeting."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pytest

import bot
import solver
import vision


# --------------------------------------------------------------------------
# Screenshot pixels -> page coordinates
# --------------------------------------------------------------------------

def test_geometry_undoes_the_device_pixel_ratio() -> None:
    """A Retina screenshot is 2x the CSS box, and must map back to CSS."""
    geometry = bot.Geometry(x=340.0, y=15.0, width=720.0, height=470.0, scale=2.0)
    assert geometry.to_page(0, 0) == (340.0, 15.0)
    assert geometry.to_page(1440, 940) == (340.0 + 720.0, 15.0 + 470.0)
    assert geometry.to_page(169.5, 179.0) == (340.0 + 84.75, 15.0 + 89.5)


def test_geometry_is_identity_at_scale_one() -> None:
    geometry = bot.Geometry(x=10.0, y=20.0, width=720.0, height=470.0, scale=1.0)
    assert geometry.to_page(100, 200) == (110.0, 220.0)


def test_drag_points_land_in_the_gaps_between_apples() -> None:
    """Start and end must sit outside the target cells but inside the neighbours' gap."""
    cal = vision.Calibration.load()
    for move in [solver.Move(0, 0, 0, 1, 2), solver.Move(3, 4, 7, 9, 5)]:
        x0, y0 = cal.corner(move.r1, move.c1)
        x1, y1 = cal.corner(move.r2 + 1, move.c2 + 1)
        # Outside the target cells...
        assert x0 < cal.centre(move.r1, move.c1)[0]
        assert x1 > cal.centre(move.r2, move.c2)[0]
        # ...but not far enough to touch the neighbouring apples.
        if move.c1 > 0:
            assert x0 > cal.centre(move.r1, move.c1 - 1)[0]
        assert y0 < cal.centre(move.r1, move.c1)[1]
        assert y1 > cal.centre(move.r2, move.c2)[1]
        # Exactly halfway between the two columns of apple centres.
        assert x0 == pytest.approx(cal.centre(0, move.c1)[0] - cal.pitch_x / 2)


# --------------------------------------------------------------------------
# Planning and execution
# --------------------------------------------------------------------------

def _args(**overrides: object) -> argparse.Namespace:
    """The CLI defaults, so tests exercise what the bot actually runs with."""
    parser = bot.build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def real_board_local() -> solver.Board:
    path = Path(__file__).parent / "fixtures" / "board_real.txt"
    return solver.parse_board(path.read_text())


@pytest.mark.parametrize("strategy", ["rollout", "greedy"])
def test_make_plan_returns_a_legal_line_to_the_end(
    real_board_local: solver.Board, strategy: str
) -> None:
    plan = bot.make_plan(real_board_local, 0.3, _args(strategy=strategy),
                         random.Random(0))
    assert plan.moves and plan.score == sum(m.apples for m in plan.moves)
    work = real_board_local.copy()
    for move in plan.moves:
        assert solver.is_legal(work, move), f"{strategy} planned illegal {move}"
        solver.apply_move(work, move, inplace=True)
    assert solver.generate_moves(work) == []     # the line plays to exhaustion


def test_plan_respects_its_budget(real_board_local: solver.Board) -> None:
    plan = solver.plan(real_board_local, 0.5, random.Random(0))
    assert plan.elapsed < 3.0
    assert plan.score > 0


def test_a_dropped_move_is_visible_at_the_next_checkpoint(
    real_board_local: solver.Board,
) -> None:
    """The checkpoint contract: skipping one drag must change the board."""
    plan = solver.plan(real_board_local, 0.2, random.Random(0))
    assert len(plan.moves) > bot.DRAG_STEPS // 4

    expected = real_board_local.copy()
    actual = real_board_local.copy()
    for i, move in enumerate(plan.moves[:10]):
        solver.apply_move(expected, move, inplace=True)
        if i != 3:                      # the game dropped move 3
            solver.apply_move(actual, move, inplace=True)

    assert not np.array_equal(actual, expected)
    assert solver.remaining_apples(actual) > solver.remaining_apples(expected)


def test_default_budgets_fit_inside_the_time_limit() -> None:
    """Planning plus dragging a whole game must leave headroom on the clock."""
    args = _args()
    typical_moves = 70          # a long game; most are shorter
    estimate = args.plan_budget + typical_moves * bot.EXECUTE_SECONDS_PER_MOVE
    assert estimate < args.time_limit
    assert args.time_limit < bot.GAME_SECONDS     # room for the score screen


def test_verify_every_is_positive() -> None:
    assert _args().verify_every >= 1
