"""SQLite storage for customers (sessions) and their try-ons. Photos live on disk next to the DB."""
from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

# On Render this is the persistent disk (DATA_DIR=/var/data); locally it's server/data.
DATA = Path(os.getenv("DATA_DIR") or Path(__file__).resolve().parent / "data")
PHOTOS = DATA / "photos"
DB_PATH = DATA / "tryon.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT,
    email TEXT,
    consent INTEGER NOT NULL DEFAULT 0,
    selfie TEXT NOT NULL,          -- file name in data/photos
    user_agent TEXT,
    ip TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tryons (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    wig_id TEXT NOT NULL,
    wig_name TEXT,
    status TEXT NOT NULL,          -- ok | error
    error TEXT,
    before_img TEXT,
    after_img TEXT,
    face_locked INTEGER,
    model TEXT,
    duration_ms INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tryons_session ON tryons(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
"""


def init():
    PHOTOS.mkdir(parents=True, exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def new_id() -> str:
    return uuid.uuid4().hex


def save_photo(data: bytes, ext: str = "jpg") -> str:
    name = f"{new_id()}.{ext}"
    (PHOTOS / name).write_bytes(data)
    return name


def create_session(name, phone, email, consent, selfie, user_agent, ip) -> str:
    sid = new_id()
    with conn() as c:
        c.execute(
            "INSERT INTO sessions (id, name, phone, email, consent, selfie, user_agent, ip, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, name, phone, email, int(consent), selfie, user_agent, ip, time.time()),
        )
    return sid


def get_session(sid: str):
    with conn() as c:
        return c.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()


def count_tryons(sid: str, status: str | None = None) -> int:
    q, args = "SELECT COUNT(*) FROM tryons WHERE session_id = ?", [sid]
    if status:
        q += " AND status = ?"
        args.append(status)
    with conn() as c:
        return c.execute(q, args).fetchone()[0]


def add_tryon(**f) -> str:
    tid = new_id()
    f = {"id": tid, "created_at": time.time(), **f}
    cols = ",".join(f)
    with conn() as c:
        c.execute(f"INSERT INTO tryons ({cols}) VALUES ({','.join('?' * len(f))})", list(f.values()))
    return tid


def delete_session(sid: str) -> bool:
    with conn() as c:
        s = c.execute("SELECT selfie FROM sessions WHERE id = ?", (sid,)).fetchone()
        if not s:
            return False
        files = [s["selfie"]] + [
            x for r in c.execute("SELECT before_img, after_img FROM tryons WHERE session_id = ?", (sid,))
            for x in (r["before_img"], r["after_img"]) if x
        ]
        c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    for name in files:
        (PHOTOS / name).unlink(missing_ok=True)
    return True
