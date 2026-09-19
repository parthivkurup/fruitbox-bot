"""Tests for the browser-independent parts of the bot: geometry and budgeting."""

from __future__ import annotations

import time

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
# Time budgeting
# --------------------------------------------------------------------------

def test_moves_remaining_is_capped_by_both_estimates() -> None:
    full = np.full((solver.ROWS, solver.COLS), 5, dtype=np.int8)
    # Early on, the whole-game move estimate is the binding one.
    assert bot.moves_remaining(full, played=0) == float(bot.EXPECTED_MOVES)
    # Late on, with few apples left, apples-remaining binds instead.
    sparse = np.zeros_like(full)
    sparse[0, :] = 5                       # one row, so solver.COLS apples left
    assert bot.moves_remaining(sparse, played=10) == pytest.approx(
        solver.COLS / bot.APPLES_PER_MOVE
    )
    # And it never drops below a small floor.
    assert bot.moves_remaining(np.zeros_like(full), played=99) == 3.0


def test_think_budget_reserves_time_for_dragging() -> None:
    board = np.full((solver.ROWS, solver.COLS), 5, dtype=np.int8)
    deadline = time.perf_counter() + 120.0
    generous = bot.think_budget(deadline, board, played=0, overhead=0.0)
    realistic = bot.think_budget(deadline, board, played=0, overhead=0.9)
    assert realistic < generous
    # 120s, minus the end reserve, minus 62 moves of overhead, over 62 moves.
    assert realistic == pytest.approx((120 - bot.END_RESERVE - 62 * 0.9) / 62, abs=0.05)


def test_think_budget_never_returns_less_than_the_floor() -> None:
    board = np.full((solver.ROWS, solver.COLS), 5, dtype=np.int8)
    expired = time.perf_counter() - 10.0
    assert bot.think_budget(expired, board, played=0, overhead=0.9, floor=0.02) == 0.02


def test_set_budget_retunes_a_live_strategy() -> None:
    strategy = solver.Rollout(time_budget=1.0)
    bot.set_budget(strategy, 0.25)
    assert strategy.time_budget == 0.25
    greedy = solver.GreedyFewest()
    bot.set_budget(greedy, 0.25)      # no time_budget attribute: must not raise


def test_make_strategy_rejects_an_unknown_name() -> None:
    assert isinstance(bot.make_strategy("rollout", 0.1, 12.0), solver.Rollout)
    assert isinstance(bot.make_strategy("greedy", 0.1, 12.0), solver.GreedyFewest)
    with pytest.raises(ValueError):
        bot.make_strategy("nope", 0.1, 12.0)
