"""catalog.json read/write (shared by the customer API and the admin wig manager)."""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# With DATA_DIR set (Render persistent disk), the catalogue and wig photos live there so admin
# changes survive redeploys. Without it, they live in the repo folder as before.
ROOT = Path(os.getenv("DATA_DIR")) if os.getenv("DATA_DIR") else REPO
CATALOG = ROOT / "catalog.json"
WIGS = ROOT / "wigs"
THUMBS = WIGS / "thumbs"


def seed() -> None:
    """First boot on an empty disk: copy the starting catalogue and wig photos from the repo."""
    if ROOT == REPO or CATALOG.exists():
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "wigs", WIGS, dirs_exist_ok=True)
    shutil.copy2(REPO / "catalog.json", CATALOG)


seed()

_lock = threading.Lock()
shapes: dict[str, dict | None] = {}  # wig id -> outline from its product photo (cache)


def load() -> dict:
    with _lock:
        return json.loads(CATALOG.read_text(encoding="utf-8"))


def save(data: dict) -> None:
    """Atomic write so a crash mid-save never leaves a broken catalogue."""
    with _lock:
        tmp = CATALOG.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CATALOG)


def forget_shape(wig_id: str) -> None:
    shapes.pop(wig_id, None)
