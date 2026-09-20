"""Pure Fruit Box game logic: board, move enumeration and playing strategies.

No browser, no I/O. A board is a (10, 17) numpy array of small ints where 0
means "cleared / empty" and 1-9 is the digit on an apple.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import NamedTuple, Protocol, Sequence

import numpy as np

ROWS: int = 10
COLS: int = 17
TARGET: int = 10

Board = np.ndarray  # shape (ROWS, COLS), integer dtype, 0 == empty


class Move(NamedTuple):
    """An axis-aligned rectangle of cells, inclusive on both ends."""

    r1: int
    c1: int
    r2: int
    c2: int
    apples: int          # non-empty cells cleared == points scored

    @property
    def height(self) -> int:
        return self.r2 - self.r1 + 1

    @property
    def width(self) -> int:
        return self.c2 - self.c1 + 1


Rect = tuple[int, int, int, int]        # (r1, c1, r2, c2), without the count


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
        weights = self.weights[apples]
        # Rescale so the best move always weighs 1: at large alpha the raw
        # weights underflow to zero and the draw would degenerate.
        weights = weights / weights.max()
        cum = np.cumsum(weights)
        return int(np.searchsorted(cum, rng.random() * cum[-1]))

    def pick(self, moves: list[Move], rng: random.Random) -> Move:
        """Same policy over an explicit move list (used by tests)."""
        if self.alpha == 0.0:
            return rng.choice(moves)
        weights = [self.weights[min(m.apples, _MAX_APPLES)] for m in moves]
        return rng.choices(moves, weights=weights, k=1)[0]


def _playout(
    board: Board,
    policy: RandomPolicy,
    rng: random.Random,
    record: list[Rect] | None = None,
) -> int:
    """Play a board to the end with ``policy``; return apples cleared.

    Apples cleared is just the drop in the apple count, so no per-move
    bookkeeping is needed unless the caller wants the sequence itself.
    """
    work = board.copy()
    before = int((work > 0).sum())
    while True:
        hits, apples = _move_arrays(work)
        if hits.size == 0:
            return before - int((work > 0).sum())
        k = policy._pick_index(hits, apples, rng)
        i, j = divmod(int(hits[k]), _NC)
        if record is not None:
            record.append((int(_R1[i]), int(_C1[j]), int(_R2[i]), int(_C2[j])))
        work[_R1[i] : _R2P1[i], _C1[j] : _C2P1[j]] = 0


def replay(board: Board, line: Sequence[Rect]) -> int | None:
    """Score a remembered line on ``board``, or None if it no longer plays."""
    work = board.copy()
    total = 0
    for r1, c1, r2, c2 in line:
        block = work[r1 : r2 + 1, c1 : c2 + 1]
        if int(block.sum()) != TARGET:
            return None
        total += int((block > 0).sum())
        block[:] = 0
    return total


@dataclass
class Rollout:
    """Main strategy: randomised greedy playouts, commit the best first move.

    Two refinements over plain round-robin rollouts:

    *Line memory* - the best playout found is kept whole. After committing its
    first move the rest is replayed as the next search's starting bid, so search
    can only ever improve on what it already found.

    *Sequential halving* - splitting the budget evenly over 50-odd candidate
    first moves wastes most of it on obvious losers. Instead the candidates are
    played in rounds and the worse half is dropped each round, concentrating the
    budget on the moves still in contention.
    """

    time_budget: float = 0.5            # per-move think time cap (s)
    total_budget: float | None = None   # whole-game think time, split per move
    max_playouts: int = 1_000_000
    alpha: float = 12.0                 # tuned; see README
    keep_line: bool = True
    halving: bool = True
    floor: float = 0.02
    name: str = "rollout"

    def __post_init__(self) -> None:
        self.policy = RandomPolicy(alpha=self.alpha)
        self.reset()

    def reset(self) -> None:
        """Start a new game: clear the line memory and the game clock."""
        self.line: list[Rect] = []
        self.played = 0
        self.playouts = 0
        self.playout_cost = 1e-3        # seconds, refined as we go
        self.deadline: float | None = (
            time.perf_counter() + self.total_budget if self.total_budget else None
        )

    # -- budgeting --------------------------------------------------------

    def _budget(self, board: Board) -> float:
        """Think time for this move, honouring any whole-game budget."""
        if self.deadline is None:
            return self.time_budget
        left = self.deadline - time.perf_counter()
        moves_left = max(3.0, min(62.0 - self.played,
                                  remaining_apples(board) / 2.1))
        return max(self.floor, min(self.time_budget, left / moves_left))

    # -- search -----------------------------------------------------------

    def choose(self, board: Board, rng: random.Random) -> Move | None:
        hits, apples = _move_arrays(board)
        n = hits.size
        if n == 0:
            return None
        self.played += 1
        if n == 1:
            self.line = []
            return _move_from_flat(int(hits[0]), int(apples[0]))

        deadline = time.perf_counter() + self._budget(board)

        # Starting bid: whatever the remembered line is still worth.
        best: list[Rect] = []
        best_score = -1
        if self.keep_line and self.line:
            carried = replay(board, self.line)
            if carried is not None:
                best, best_score = list(self.line), carried

        # A playout costs about a millisecond. If the budget cannot cover one
        # per candidate there is nothing to halve, so fall back to trying the
        # smallest clears - the same bias the rollout policy uses.
        affordable = max(1, int((deadline - time.perf_counter()) / self.playout_cost))
        if self.halving and affordable >= n:
            alive = list(range(n))
            rounds = max(1, (n - 1).bit_length())
        else:
            alive = [int(k) for k in np.argsort(apples, kind="stable")[:affordable]]
            rounds = 1

        scores: dict[int, int] = {k: -1 for k in alive}
        lines: dict[int, list[Rect]] = {}
        children: dict[int, Board] = {}
        for round_index in range(rounds):
            share = (deadline - time.perf_counter()) / (rounds - round_index)
            stop = time.perf_counter() + share
            while True:
                started = time.perf_counter()
                for k in alive:
                    if k not in children:
                        children[k] = self._child(board, hits, k)
                    record: list[Rect] = []
                    total = int(apples[k]) + _playout(
                        children[k], self.policy, rng, record
                    )
                    self.playouts += 1
                    if total > scores[k]:
                        scores[k] = total
                        lines[k] = [self._rect(hits, k), *record]
                    if total > best_score:
                        best_score, best = total, lines[k]
                now = time.perf_counter()
                self.playout_cost = (0.9 * self.playout_cost
                                     + 0.1 * (now - started) / len(alive))
                if now >= stop or self.playouts >= self.max_playouts:
                    break
            if len(alive) > 1:
                alive.sort(key=lambda k: -scores[k])
                alive = alive[: max(1, len(alive) // 2)]

        if not best:
            best = lines.get(alive[0], [self._rect(hits, alive[0])])
        self.line = best[1:] if self.keep_line else []
        r1, c1, r2, c2 = best[0]
        return Move(r1, c1, r2, c2, count_apples(board, Move(r1, c1, r2, c2, 0)))

    @staticmethod
    def _rect(hits: np.ndarray, k: int) -> Rect:
        i, j = divmod(int(hits[k]), _NC)
        return int(_R1[i]), int(_C1[j]), int(_R2[i]), int(_C2[j])

    @staticmethod
    def _child(board: Board, hits: np.ndarray, k: int) -> Board:
        i, j = divmod(int(hits[k]), _NC)
        child = board.copy()
        child[_R1[i] : _R2P1[i], _C1[j] : _C2P1[j]] = 0
        return child


@dataclass
class Beam:
    """Beam search over whole-game lines, committing the best first move.

    Kept for comparison only. Ranking partial lines by apples cleared so far
    rewards big early clears, which is exactly the wrong instinct; adding a
    mobility term (how many moves the position still offers) helps a lot but
    beam still loses to plain greedy, let alone to rollouts. See the README.
    """

    width: int = 24
    time_budget: float = 1.0
    mobility: float = 0.3      # weight on moves still available
    name: str = "beam"

    def _rank(self, entry: tuple[int, "Board", Move]) -> float:
        return -(entry[0] + self.mobility * count_moves(entry[1]))

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
        beam.sort(key=self._rank)
        beam = beam[: self.width]
        best = max(beam, key=lambda e: e[0])
        best_score, best_move = best[0], best[2]

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
            children.sort(key=self._rank)
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
        return self.elapsed / len(self.moves) if self.moves else 0.0


@dataclass
class Plan:
    """A complete line of play worked out ahead of time."""

    moves: list[Move]
    score: int
    elapsed: float

    def __len__(self) -> int:
        return len(self.moves)


def plan(
    board: Board,
    budget: float,
    rng: random.Random | None = None,
    *,
    alpha: float = 12.0,
) -> Plan:
    """Search for ``budget`` seconds and return a whole move sequence.

    This is the same rollout search the move-at-a-time strategy uses, run to
    the end of the game in one go: the search spends its budget across the
    moves it expects to make and hands back the line it settled on.
    """
    strategy = Rollout(time_budget=max(budget, 0.05), total_budget=budget, alpha=alpha)
    result = simulate(board, strategy, rng)
    return Plan(moves=result.moves, score=result.score, elapsed=result.elapsed)


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
    reset = getattr(strategy, "reset", None)
    if callable(reset):
        reset()
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
