"""Fail with a useful message when the virtualenv is not active.

The entry points import this before anything third-party, so running them with
the system interpreter says what to do instead of raising a bare
``ModuleNotFoundError`` on whichever dependency happens to be imported first.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REQUIRED = ("numpy", "cv2", "playwright")


def check(required: tuple[str, ...] = REQUIRED) -> None:
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return
    root = Path(__file__).resolve().parent
    raise SystemExit(
        f"missing dependencies: {', '.join(missing)}\n"
        f"(running {sys.executable})\n\n"
        f"fruitbox-bot keeps its dependencies in a virtualenv. From {root}:\n"
        f"    source .venv/bin/activate\n"
        f"    pip install -r requirements.txt && playwright install chromium\n\n"
        f"or run it without activating:\n"
        f"    {root / '.venv/bin/python'} {Path(sys.argv[0]).name or 'bot.py'}"
    )


check()
