"""Pure Fruit Box game logic: board, move enumeration and playing strategies.

No browser, no I/O. A board is a (10, 17) numpy array of small ints where 0
means "cleared / empty" and 1-9 is the digit on an apple.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

ROWS: int = 10
COLS: int = 17
TARGET: int = 10

Board = np.ndarray  # shape (ROWS, COLS), integer dtype, 0 == empty


class Move(Sequence):
    """An axis-aligned rectangle of cells, inclusive on both ends."""

    __slots__ = ("r1", "c1", "r2", "c2", "apples")

    def __init__(self, r1: int, c1: int, r2: int, c2: int, apples: int) -> None:
        self.r1 = r1
        self.c1 = c1
        self.r2 = r2
        self.c2 = c2
        self.apples = apples  # non-empty cells cleared == points scored

    def __getitem__(self, i: int):  # lets a Move unpack like a tuple
        return (self.r1, self.c1, self.r2, self.c2, self.apples)[i]

    def __len__(self) -> int:
        return 5

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Move):
            return tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return tuple(self) == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        return (
            f"Move(r{self.r1}-{self.r2}, c{self.c1}-{self.c2}, "
            f"{self.apples} apples)"
        )

    @property
    def height(self) -> int:
        return self.r2 - self.r1 + 1

    @property
    def width(self) -> int:
        return self.c2 - self.c1 + 1


# --------------------------------------------------------------------------
# Rectangle enumeration via 2D prefix sums
# --------------------------------------------------------------------------

def _rect_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """All (start, end) index pairs with start <= end, as two flat arrays."""
    starts, ends = np.triu_indices(n)
    return starts.astype(np.intp), ends.astype(np.intp)


_R1, _R2 = _rect_pairs(ROWS)
_C1, _C2 = _rect_pairs(COLS)
_R2P1 = _R2 + 1
_C2P1 = _C2 + 1
_NR = _R1.size          # 55 row spans
_NC = _C1.size          # 153 column spans

# Cells are encoded as ``digit + 4096 * (digit > 0)`` so a single prefix sum
# yields both the digit total (low 12 bits, max 9*170 = 1530) and the apple
# count (high bits) for every rectangle at once.
_SHIFT = 12
_MASK = (1 << _SHIFT) - 1

# Weight lookup for the randomised policy: a legal move clears 2..10 apples.
_MAX_APPLES = TARGET + 1


def prefix_sum(board: Board) -> np.ndarray:
    """Inclusive 2D prefix sum padded with a zero row and column."""
    out = np.zeros((board.shape[0] + 1, board.shape[1] + 1), dtype=np.int32)
    np.cumsum(np.cumsum(board, axis=0, dtype=np.int32), axis=1, out=out[1:, 1:])
    return out


def _encode(board: Board) -> np.ndarray:
    enc = board.astype(np.int32)
    enc += (enc > 0) << _SHIFT
    return enc


def rect_totals(board: Board) -> np.ndarray:
    """Encoded total of every rectangle, shaped (_NR, _NC).

    Row spans index the first axis, column spans the second; decode with
    ``& _MASK`` for the digit sum and ``>> _SHIFT`` for the apple count.
    """
    enc = _encode(board)
    rows = np.zeros((ROWS + 1, COLS), dtype=np.int32)
    np.cumsum(enc, axis=0, out=rows[1:])
    band = rows[_R2P1] - rows[_R1]                  # (_NR, COLS)
    cols = np.zeros((_NR, COLS + 1), dtype=np.int32)
    np.cumsum(band, axis=1, out=cols[:, 1:])
    return cols[:, _C2P1] - cols[:, _C1]            # (_NR, _NC)


def _move_arrays(board: Board) -> tuple[np.ndarray, np.ndarray]:
    """Flat rectangle indices summing to TARGET, plus their apple counts."""
    flat = rect_totals(board).ravel()
    hits = np.flatnonzero((flat & _MASK) == TARGET)
    return hits, flat[hits] >> _SHIFT


def _move_from_flat(index: int, apples: int) -> Move:
    i, j = divmod(index, _NC)
    return Move(int(_R1[i]), int(_C1[j]), int(_R2[i]), int(_C2[j]), int(apples))


def generate_moves(board: Board) -> list[Move]:
    """Every rectangle whose digits sum to exactly TARGET."""
    hits, apples = _move_arrays(board)
    return [_move_from_flat(int(h), int(a)) for h, a in zip(hits, apples)]


def count_moves(board: Board) -> int:
    """Number of legal moves, without building Move objects."""
    return int(_move_arrays(board)[0].size)


def is_legal(board: Board, move: Move) -> bool:
    """True if the rectangle sums to TARGET and contains at least one apple."""
    if not (0 <= move.r1 <= move.r2 < board.shape[0]):
        return False
    if not (0 <= move.c1 <= move.c2 < board.shape[1]):
        return False
    block = board[move.r1 : move.r2 + 1, move.c1 : move.c2 + 1]
    return int(block.sum()) == TARGET and bool((block > 0).any())


def apply_move(board: Board, move: Move, *, inplace: bool = False) -> Board:
    """Clear the rectangle, returning the resulting board."""
    out = board if inplace else board.copy()
    out[move.r1 : move.r2 + 1, move.c1 : move.c2 + 1] = 0
    return out


def count_apples(board: Board, move: Move) -> int:
    block = board[move.r1 : move.r2 + 1, move.c1 : move.c2 + 1]
    return int((block > 0).sum())


# --------------------------------------------------------------------------
# Board helpers
# --------------------------------------------------------------------------

def parse_board(text: str) -> Board:
    """Read a board from whitespace-separated digits (10 rows x 17 columns)."""
    rows = [line.split() for line in text.strip().splitlines() if line.strip()]
    arr = np.array([[int(v) for v in row] for row in rows], dtype=np.int8)
    if arr.shape != (ROWS, COLS):
        raise ValueError(f"expected {(ROWS, COLS)} board, got {arr.shape}")
    return arr


def board_to_str(board: Board, empty: str = ".") -> str:
    return "\n".join(
        " ".join(empty if v == 0 else str(int(v)) for v in row) for row in board
    )


def random_board(rng: random.Random | np.random.Generator | None = None) -> Board:
    """A fresh board of uniform random digits 1-9, like the real game deals."""
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        seed = rng.randrange(2**32) if isinstance(rng, random.Random) else None
        gen = np.random.default_rng(seed)
    return gen.integers(1, 10, size=(ROWS, COLS), dtype=np.int8)


def remaining_apples(board: Board) -> int:
    return int((board > 0).sum())


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

class Strategy(Protocol):
    name: str

    def choose(self, board: Board, rng: random.Random) -> Move | None:
        """Pick the next move, or None to stop."""


@dataclass
class GreedyFewest:
    """Baseline: always clear the fewest apples possible."""

    name: str = "greedy-fewest"

    def choose(self, board: Board, rng: random.Random) -> Move | None:
        moves = generate_moves(board)
        if not moves:
            return None
        return min(moves, key=lambda m: (m.apples, m.r1, m.c1, m.r2, m.c2))


@dataclass
class GreedyMost:
    """Control: always clear the most apples possible."""

    name: str = "greedy-most"

    def choose(self, board: Board, rng: random.Random) -> Move | None:
        moves = generate_moves(board)
        if not moves:
            return None
        return max(moves, key=lambda m: (m.apples, -m.r1, -m.c1))


@dataclass
class RandomPolicy:
    """Random move, biased towards clearing few apples.

    ``alpha`` = 0 is uniform; larger values approach greedy-fewest.
    """

    alpha: float = 2.0
    name: str = "random-policy"

    def __post_init__(self) -> None:
        counts = np.arange(_MAX_APPLES + 1, dtype=np.float64)
        counts[0] = 1.0
        self.weights: np.ndarray = counts**-self.alpha if self.alpha else np.ones_like(counts)

    def choose(self, board: Board, rng: random.Random) -> Move | None:
        hits, apples = _move_arrays(board)
        if hits.size == 0:
            return None
        k = self._pick_index(hits, apples, rng)
        return _move_from_flat(int(hits[k]), int(apples[k]))

    def _pick_index(
        self, hits: np.ndarray, apples: np.ndarray, rng: random.Random
    ) -> int:
        if hits.size == 1:
            return 0
        if self.alpha == 0.0:
            return rng.randrange(hits.size)
        cum = np.cumsum(self.weights[apples])
        return int(np.searchsorted(cum, rng.random() * cum[-1]))

    def pick(self, moves: list[Move], rng: random.Random) -> Move:
        """Same policy over an explicit move list (used by tests)."""
        if self.alpha == 0.0:
            return rng.choice(moves)
        weights = [self.weights[min(m.apples, _MAX_APPLES)] for m in moves]
        return rng.choices(moves, weights=weights, k=1)[0]


def _playout(board: Board, policy: RandomPolicy, rng: random.Random) -> int:
    """Play a board to the end with ``policy``; return apples cleared."""
    work = board.copy()
    before = int((work > 0).sum())
    while True:
        hits, apples = _move_arrays(work)
        if hits.size == 0:
            return before - int((work > 0).sum())
        k = policy._pick_index(hits, apples, rng)
        i, j = divmod(int(hits[k]), _NC)
        work[_R1[i] : _R2P1[i], _C1[j] : _C2P1[j]] = 0


@dataclass
class Rollout:
    """Main strategy: randomised greedy playouts, commit the best first move.

    Each iteration picks a candidate first move (round-robin over all legal
    moves), plays the rest of the game out with ``policy``, and keeps the first
    move of the highest-scoring playout seen.
    """

    time_budget: float = 0.5          # seconds per move
    max_playouts: int = 1_000_000
    min_playouts: int = 1             # at least this many rounds per candidate
    alpha: float = 2.0
    name: str = "rollout"

    def __post_init__(self) -> None:
        self.policy = RandomPolicy(alpha=self.alpha)
        self.last_playouts: int = 0

    def choose(self, board: Board, rng: random.Random) -> Move | None:
        hits, apples = _move_arrays(board)
        n = hits.size
        if n == 0:
            return None
        if n == 1:
            return _move_from_flat(int(hits[0]), int(apples[0]))

        deadline = time.perf_counter() + self.time_budget
        best_index = 0
        best_score = -1
        playouts = 0
        floor = self.min_playouts * n
        while playouts < self.max_playouts:
            if playouts >= floor and time.perf_counter() >= deadline:
                break
            k = playouts % n
            playouts += 1
            i, j = divmod(int(hits[k]), _NC)
            child = board.copy()
            child[_R1[i] : _R2P1[i], _C1[j] : _C2P1[j]] = 0
            total = int(apples[k]) + _playout(child, self.policy, rng)
            if total > best_score:
                best_score, best_index = total, k
        self.last_playouts = playouts
        return _move_from_flat(int(hits[best_index]), int(apples[best_index]))


@dataclass
class Beam:
    """Beam search over whole-game lines, committing the best first move."""

    width: int = 24
    time_budget: float = 1.0
    name: str = "beam"

    def choose(self, board: Board, rng: random.Random) -> Move | None:
        moves = generate_moves(board)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        deadline = time.perf_counter() + self.time_budget
        # Each beam entry: (score so far, board, first move taken)
        beam: list[tuple[int, Board, Move]] = [
            (m.apples, apply_move(board, m), m) for m in moves
        ]
        beam.sort(key=lambda e: -e[0])
        beam = beam[: self.width]
        best_score, best_move = beam[0][0], beam[0][2]

        while beam and time.perf_counter() < deadline:
            children: list[tuple[int, Board, Move]] = []
            seen: set[bytes] = set()
            for score, state, first in beam:
                for m in generate_moves(state):
                    child = apply_move(state, m)
                    key = child.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    children.append((score + m.apples, child, first))
            if not children:
                break
            children.sort(key=lambda e: -e[0])
            beam = children[: self.width]
            if beam[0][0] > best_score:
                best_score, best_move = beam[0][0], beam[0][2]
        return best_move


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class SimResult:
    score: int
    moves: list[Move]
    elapsed: float

    @property
    def move_count(self) -> int:
        return len(self.moves)

    @property
    def time_per_move(self) -> float:
        return self.elapsed / self.moves.__len__() if self.moves else 0.0


def simulate(
    board: Board,
    strategy: Strategy,
    rng: random.Random | None = None,
    *,
    verify: bool = True,
    time_limit: float | None = None,
) -> SimResult:
    """Play ``board`` to the end with ``strategy`` and report the final score."""
    rng = rng or random.Random(0)
    work = board.copy()
    played: list[Move] = []
    score = 0
    started = time.perf_counter()
    while True:
        if time_limit is not None and time.perf_counter() - started >= time_limit:
            break
        move = strategy.choose(work, rng)
        if move is None:
            break
        if verify and not is_legal(work, move):
            raise ValueError(f"strategy {strategy.name} produced illegal {move!r}")
        move = Move(move.r1, move.c1, move.r2, move.c2, count_apples(work, move))
        apply_move(work, move, inplace=True)
        played.append(move)
        score += move.apples
    return SimResult(score=score, moves=played, elapsed=time.perf_counter() - started)
