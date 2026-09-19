"""Unit tests for the pure game logic."""

from __future__ import annotations

import random

import numpy as np
import pytest

import solver
from solver import COLS, ROWS, TARGET, Move


def brute_force_moves(board: solver.Board) -> set[tuple[int, int, int, int, int]]:
    """Reference implementation: check every rectangle with plain loops."""
    found = set()
    for r1 in range(ROWS):
        for r2 in range(r1, ROWS):
            for c1 in range(COLS):
                for c2 in range(c1, COLS):
                    block = board[r1 : r2 + 1, c1 : c2 + 1]
                    if int(block.sum()) == TARGET and int((block > 0).sum()) >= 1:
                        found.add((r1, c1, r2, c2, int((block > 0).sum())))
    return found


# --------------------------------------------------------------------------
# Prefix sums
# --------------------------------------------------------------------------

def test_prefix_sum_matches_naive_sums() -> None:
    rng = np.random.default_rng(7)
    board = rng.integers(0, 10, size=(ROWS, COLS), dtype=np.int8)
    prefix = solver.prefix_sum(board)
    assert prefix.shape == (ROWS + 1, COLS + 1)
    assert not prefix[0].any() and not prefix[:, 0].any()
    for r1, r2, c1, c2 in [(0, 0, 0, 0), (0, 9, 0, 16), (3, 7, 2, 11), (5, 5, 4, 4)]:
        expected = int(board[r1 : r2 + 1, c1 : c2 + 1].sum())
        got = (
            prefix[r2 + 1, c2 + 1]
            - prefix[r1, c2 + 1]
            - prefix[r2 + 1, c1]
            + prefix[r1, c1]
        )
        assert int(got) == expected


def test_rect_totals_encodes_sum_and_count() -> None:
    rng = np.random.default_rng(11)
    board = rng.integers(0, 10, size=(ROWS, COLS), dtype=np.int8)
    totals = solver.rect_totals(board)
    assert totals.shape == (solver._NR, solver._NC)
    for i in (0, 12, 40, solver._NR - 1):
        for j in (0, 33, 100, solver._NC - 1):
            r1, r2 = solver._R1[i], solver._R2[i]
            c1, c2 = solver._C1[j], solver._C2[j]
            block = board[r1 : r2 + 1, c1 : c2 + 1]
            assert int(totals[i, j] & solver._MASK) == int(block.sum())
            assert int(totals[i, j] >> solver._SHIFT) == int((block > 0).sum())


# --------------------------------------------------------------------------
# Move generation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_generate_moves_matches_brute_force(seed: int) -> None:
    board = solver.random_board(np.random.default_rng(seed))
    assert {tuple(m) for m in solver.generate_moves(board)} == brute_force_moves(board)


def test_generate_moves_on_sparse_board_matches_brute_force() -> None:
    """Boards with many cleared cells are where span-over-gaps bugs show up."""
    rng = np.random.default_rng(3)
    board = solver.random_board(rng)
    board[rng.random((ROWS, COLS)) < 0.6] = 0
    assert {tuple(m) for m in solver.generate_moves(board)} == brute_force_moves(board)


def test_moves_span_empty_cells() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[0, 0] = 4
    board[0, 4] = 6  # three cleared cells in between
    moves = {tuple(m) for m in solver.generate_moves(board)}
    assert (0, 0, 0, 4, 2) in moves
    assert solver.is_legal(board, Move(0, 0, 0, 4, 2))
    # A taller rectangle over the same apples plus empty rows is also legal.
    assert solver.is_legal(board, Move(0, 0, 3, 4, 2))


def test_empty_rectangle_is_not_a_move() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    assert solver.generate_moves(board) == []
    assert not solver.is_legal(board, Move(0, 0, 2, 2, 0))


def test_is_legal_rejects_wrong_sum_and_out_of_bounds() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[0, 0] = 4
    board[0, 1] = 5
    assert not solver.is_legal(board, Move(0, 0, 0, 1, 2))  # sums to 9
    assert not solver.is_legal(board, Move(0, 0, 0, COLS, 2))
    assert not solver.is_legal(board, Move(-1, 0, 0, 1, 2))
    assert not solver.is_legal(board, Move(5, 0, 2, 1, 2))  # r1 > r2


def test_apples_count_ignores_cleared_cells() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[2, 3] = 7
    board[2, 6] = 3
    moves = solver.generate_moves(board)
    assert moves, "7 + 3 over a gap must be playable"
    # Every rectangle that reaches TARGET holds exactly these two apples,
    # however much empty space it also covers.
    assert {m.apples for m in moves} == {2}
    tight = Move(2, 3, 2, 6, 2)
    assert tight in moves and tight.width == 4 and tight.height == 1


def test_apply_move_clears_only_the_rectangle() -> None:
    board = solver.random_board(np.random.default_rng(2))
    before = board.copy()
    move = Move(1, 2, 3, 5, 0)
    after = solver.apply_move(board, move)
    assert np.array_equal(board, before)  # not mutated
    assert not after[1:4, 2:6].any()
    mask = np.ones_like(after, dtype=bool)
    mask[1:4, 2:6] = False
    assert np.array_equal(after[mask], before[mask])


# --------------------------------------------------------------------------
# Board helpers
# --------------------------------------------------------------------------

def test_parse_board_round_trip(real_board: solver.Board) -> None:
    assert real_board.shape == (ROWS, COLS)
    assert solver.parse_board(solver.board_to_str(real_board)).tolist() == real_board.tolist()
    assert real_board[0, 0] == 4 and real_board[9, 16] == 2
    assert solver.remaining_apples(real_board) == ROWS * COLS


def test_parse_board_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        solver.parse_board("1 2 3\n4 5 6")


def test_board_to_str_marks_empties() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[0, 0] = 5
    assert solver.board_to_str(board).splitlines()[0].startswith("5 . .")


# --------------------------------------------------------------------------
# Strategies and simulation
# --------------------------------------------------------------------------

STRATEGIES = [
    solver.GreedyFewest(),
    solver.GreedyMost(),
    solver.RandomPolicy(alpha=2.0),
    solver.Rollout(time_budget=0.02),
    solver.Beam(width=4, time_budget=0.05),
]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_simulate_never_makes_an_illegal_move(strategy: solver.Strategy) -> None:
    """Replay every move against the board it was played on."""
    board = solver.random_board(np.random.default_rng(5))
    result = solver.simulate(board, strategy, random.Random(0))
    work = board.copy()
    for move in result.moves:
        assert solver.is_legal(work, move), f"{strategy.name} played illegal {move}"
        assert move.apples == solver.count_apples(work, move)
        solver.apply_move(work, move, inplace=True)
    assert result.score == sum(m.apples for m in result.moves)
    assert result.score == solver.remaining_apples(board) - solver.remaining_apples(work)
    assert solver.generate_moves(work) == []  # played to exhaustion


def test_simulate_detects_an_illegal_strategy() -> None:
    class Cheater:
        name = "cheater"

        def choose(self, board: solver.Board, rng: random.Random) -> Move:
            return Move(0, 0, 0, 0, 1)

    board = np.full((ROWS, COLS), 9, dtype=np.int8)
    with pytest.raises(ValueError, match="illegal"):
        solver.simulate(board, Cheater(), random.Random(0))


def test_simulate_on_exhausted_board_scores_zero() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    result = solver.simulate(board, solver.GreedyFewest())
    assert result.score == 0 and result.moves == []


def test_greedy_fewest_picks_the_smallest_clear() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[0, 0] = 1
    board[0, 1] = 2
    board[0, 2] = 3
    board[0, 3] = 4  # 1+2+3+4 == 10, four apples
    board[5, 10] = 9
    board[5, 11] = 1  # 9+1 == 10, two apples
    move = solver.GreedyFewest().choose(board, random.Random(0))
    assert move is not None and move.apples == 2
    # It must be the 9/1 pair, not the four-apple run.
    assert move.c1 <= 10 and move.c2 >= 11 and move.r1 <= 5 <= move.r2
    assert move.c1 > 3  # the 1-2-3-4 run is untouched


def test_score_is_bounded_by_the_board(real_board: solver.Board) -> None:
    result = solver.simulate(real_board, solver.GreedyFewest())
    assert 0 < result.score <= ROWS * COLS


def test_rollout_scores_on_the_real_board(real_board: solver.Board) -> None:
    strategy = solver.Rollout(time_budget=0.05)
    assert solver.simulate(real_board, strategy, random.Random(42)).score > 0


def test_rollout_reset_clears_line_memory(real_board: solver.Board) -> None:
    strategy = solver.Rollout(time_budget=0.05)
    solver.simulate(real_board, strategy, random.Random(1))
    assert strategy.played > 0
    strategy.reset()
    assert strategy.line == [] and strategy.played == 0


def test_rollout_respects_a_whole_game_budget(real_board: solver.Board) -> None:
    strategy = solver.Rollout(time_budget=5.0, total_budget=1.0)
    result = solver.simulate(real_board, strategy, random.Random(1))
    assert result.elapsed < 3.0     # budget plus the last move's overshoot
    assert result.score > 0


def test_replay_scores_a_line_and_rejects_a_stale_one() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[0, 0], board[0, 1] = 4, 6
    board[5, 5], board[5, 6] = 3, 7
    assert solver.replay(board, [(0, 0, 0, 1), (5, 5, 5, 6)]) == 4
    assert solver.replay(board, [(0, 0, 0, 1), (0, 0, 0, 1)]) is None
    assert solver.replay(board, [(0, 0, 0, 0)]) is None


def test_remembered_line_is_actually_playable(real_board: solver.Board) -> None:
    """Whatever the strategy carries forward must replay on the live board."""
    strategy = solver.Rollout(time_budget=0.05)
    rng = random.Random(3)
    strategy.reset()
    board = real_board.copy()
    for _ in range(6):
        move = strategy.choose(board, rng)
        assert move is not None and solver.is_legal(board, move)
        solver.apply_move(board, move, inplace=True)
        if strategy.line:
            assert solver.replay(board, strategy.line) is not None
