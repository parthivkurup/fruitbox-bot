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
| `tests/` | pytest suite for the solver. |

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
python benchmark.py              # compare strategies
python -m pytest                 # unit tests
```

## Calibrating

Grid geometry and the digit templates live in `calibration.json` and
`templates/`. They only need redoing if the game's layout or colours change (for
example after toggling "Light Colors", or on a different display scale).

```bash
python calibrate.py              # opens the game, detects the grid, saves templates
```

`calibrate.py` walks the detected cells, shows you what it thinks each digit is,
and writes `templates/<digit>.png` for the ones you confirm. Re-run
`python bot.py --dry-run` afterwards and check the printed grid against the
screen before playing for real.
