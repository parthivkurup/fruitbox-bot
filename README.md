# fruitbox-bot

A Python bot that plays [Fruit Box](https://en.gamesaien.com/game/fruit_box/) by GameSaien.

Fruit Box deals a 17x10 grid of apples labelled 1-9. You drag rectangles; if the
digits inside a rectangle sum to exactly 10 the apples clear and you score one
point per apple. Cleared cells count as 0, so later rectangles can span the gaps.
120 seconds, 170 points maximum.

The bot reads the board off the game canvas with OpenCV template matching,
searches for a good line of play, and drags the rectangles with Playwright.

## Components

| File | Purpose |
| --- | --- |
| `solver.py` | Pure logic. Board is a 10x17 numpy array (0 = empty). Rectangle enumeration by 2D prefix sum, plus the playing strategies. |
| `vision.py` | Reads a board from a canvas screenshot: grid calibration, digit templates, empty-cell detection. |
| `calibrate.py` | Interactive helper: grabs a screenshot, finds the grid, and saves one template per digit. |
| `bot.py` | Playwright driver: opens the game, starts it, reads, plans, drags, repeats. |
| `benchmark.py` | Scores each strategy over the fixture board and N random boards. |
| `tests/` | pytest suite: solver logic, plus a vision regression test against a saved canvas screenshot. |

## Strategy

Scores are apples cleared, out of 170.

| strategy | what it does | mean |
| --- | --- | --- |
| greedy-most | clears the biggest rectangle it can | 98 |
| random-uniform | any legal move | 103 |
| beam (width 24) | beam search over whole-game lines | 109 |
| random-biased | random, weighted towards small clears | 113 |
| greedy-fewest | always the smallest clear | 115 |
| rollout (plain) | randomised playouts, best first move | 127 |
| **rollout (tuned)** | **plus line memory and sequential halving** | **135** |

(200 random boards, 50ms of thinking per move.)

Given the thinking time a real game actually affords - about 60 seconds, once
dragging and re-reading are paid for - over 60 random boards plus the fixture:

| strategy | mean | median | min | max | fixture |
| --- | --- | --- | --- | --- | --- |
| greedy-fewest | 112.0 | 112 | 74 | 134 | 115 |
| rollout (plain) | 126.7 | 126 | 89 | 151 | 140 |
| **rollout (tuned)** | **135.7** | 136 | 96 | 167 | **150** |

Live games land in the same place: the last headed run cleared 137 of 170 in
118 of the 120 seconds, finishing because the board ran dry rather than because
the clock did.

Clearing few apples at a time is the whole game: a two-apple clear opens gaps
that let later rectangles reach across the board, whereas a big clear spends
several apples to buy one move's worth of progress. Every strategy that scores
well is some refinement of "prefer small clears".

The rollout strategy plays the position out to the end many times with a
randomised version of that preference, and commits the first move of the best
playout it saw. Two things were worth adding on top:

- **Line memory** (+4 mean). The best playout is remembered whole. After its
  first move is committed the rest is replayed as the next search's opening bid,
  so a good line found once is never lost to an unlucky later search.
- **Sequential halving** (neutral, kept on). The budget is spent in rounds, and
  the worse half of the candidate first moves is dropped after each, rather than
  spreading it evenly over fifty-odd candidates.

Tuning the policy's bias towards small clears (`--alpha`) mattered more than
either: it climbs steadily from alpha 1 to about 12, where it plateaus. At the
limit the policy is "pick uniformly among the smallest clears available".

Beam search is kept in `benchmark.py` as a control. Ranking partial lines by
apples cleared so far is a bad signal - it rewards exactly the greedy-most
instinct that loses - and even with a mobility term it does not catch greedy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Running

```bash
python bot.py --dry-run          # read the board, print the plan, drag nothing
python bot.py                    # play for real
python bot.py --debug            # also write annotated screenshots to debug/
python bot.py --headless         # no visible window (useful for testing)
python benchmark.py              # compare strategies, 200 random boards
python benchmark.py --boards 60 --total-budget 60   # as the live bot plays
python -m pytest                 # unit tests
```

The bot budgets its own thinking: it measures how long dragging and re-reading
actually cost, and splits whatever is left of the 120 seconds across the moves
it still expects to make. A game typically runs 60 moves, spends about 55
seconds on browser work and 60 on search, and finishes when the board runs dry
with a few seconds to spare.

### Playing the board reliably

Three things about the browser side are worth knowing, because they were all
found the hard way:

- The game samples the pointer once per animation frame, so a drag made of a
  few big jumps is often dropped entirely - long thin rectangles worst of all.
  Drags use 12 intermediate positions, and a dropped drag is retried with 28.
- A clear resolves within about 300ms. If the board is still untouched after
  400ms the drag was dropped, so the bot retries rather than waiting.
- Canvas captures use JPEG, which is about 3x faster than PNG (33ms vs 92ms)
  and parses to the same board. Calibration uses PNG.

The bot keeps its own model of the board and re-reads the screen to stay honest.
After every move it counts apples (cheap - no digit matching) to confirm the
clear happened, and re-parses the digits on the interval set by
`--reread-every`, adopting what the screen says over what it expected. If any
digit matches weakly it keeps the tracked board instead, so one bad read cannot
derail the game.

## Calibrating

Grid geometry and the digit templates live in `calibration.json` and
`templates/`. They only need redoing if the game's layout or colours change (for
example after toggling "Light Colors", or on a different display scale).

```bash
python calibrate.py              # opens the game, detects the grid, saves templates
```

It opens the game, locates the grid by projecting the mask of red apples (17
bands across, 10 down), then crops all 170 cells and groups the digit glyphs
into nine look-alike clusters. It prints them as ASCII art and asks you to read
them off in order:

```
cluster:
     1            2            3    ...
.#########.  ..#######..  ..#######..
.###.......  .###...###.  .###..####.
...
type the nine digits above, left to right (e.g. 598671243):
```

Answer with the nine digits and it writes `templates/<digit>.png` plus
`calibration.json`, then reads the board back so you can check it. There is also
a contact sheet at `templates/_clusters.png` if the ASCII is hard to read.

```bash
python calibrate.py --image board.png     # from a saved screenshot
python calibrate.py --labels 598671243    # skip the prompt
python calibrate.py --save-cells cells/   # also dump every cell crop
python calibrate.py --dry-run             # report, write nothing
```

Afterwards run `python bot.py --dry-run` and check the printed grid against the
screen before playing for real.
