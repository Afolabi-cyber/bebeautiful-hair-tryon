"""Admin API: who took pictures, which wigs they tried, stats, CSV export. Password from ADMIN_PASSWORD."""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import os
import time
from datetime import datetime, timedelta

import re
import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from PIL import Image, ImageOps
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import catalog_store
import db
from status import STATUS

router = APIRouter(prefix="/api/admin")
TOKEN_HOURS = 12


def _secret() -> bytes:
    pw = os.getenv("ADMIN_PASSWORD", "")
    if not pw:
        raise HTTPException(503, "Admin is disabled. Set ADMIN_PASSWORD in .env")
    return (pw + os.getenv("SECRET_KEY", "change-me")).encode()


def make_token() -> str:
    exp = int(time.time()) + TOKEN_HOURS * 3600
    sig = hmac.new(_secret(), str(exp).encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def check_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    exp, sig = token.split(".", 1)
    good = hmac.new(_secret(), exp.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, good) and exp.isdigit() and int(exp) > time.time()


def require_admin(authorization: str | None = Header(None), t: str | None = Query(None)):
    token = (authorization or "").removeprefix("Bearer ").strip() or t
    if not check_token(token):
        raise HTTPException(401, "Please sign in again")
    return token


class Login(BaseModel):
    password: str


@router.post("/login")
def login(body: Login):
    _secret()
    if not hmac.compare_digest(body.password, os.getenv("ADMIN_PASSWORD", "")):
        time.sleep(1)  # slow down guessing
        raise HTTPException(401, "Wrong password")
    return {"token": make_token(), "expiresInHours": TOKEN_HOURS}


def media_url(name: str | None, token: str) -> str | None:
    return f"/api/admin/media/{name}?t={token}" if name else None


@router.get("/media/{name}")
def media(name: str, _: str = Depends(require_admin)):
    path = (db.PHOTOS / name).resolve()
    if db.PHOTOS.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/stats")
def stats(_: str = Depends(require_admin)):
    start_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    since = (datetime.now() - timedelta(days=13)).replace(hour=0, minute=0, second=0, microsecond=0)
    with db.conn() as c:
        people = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        people_today = c.execute("SELECT COUNT(*) FROM sessions WHERE created_at >= ?", (start_today,)).fetchone()[0]
        tries = c.execute("SELECT COUNT(*) FROM tryons").fetchone()[0]
        ok = c.execute("SELECT COUNT(*) FROM tryons WHERE status = 'ok'").fetchone()[0]
        tries_today = c.execute("SELECT COUNT(*) FROM tryons WHERE created_at >= ?", (start_today,)).fetchone()[0]
        avg = c.execute("SELECT AVG(duration_ms) FROM tryons WHERE status = 'ok'").fetchone()[0]
        wigs = [dict(r) for r in c.execute(
            "SELECT wig_id, wig_name, COUNT(*) AS count FROM tryons WHERE status = 'ok' "
            "GROUP BY wig_id ORDER BY count DESC")]
        rows = c.execute("SELECT created_at FROM tryons WHERE created_at >= ?", (since.timestamp(),)).fetchall()
    days = {(since + timedelta(days=i)).strftime("%Y-%m-%d"): 0 for i in range(14)}
    for r in rows:
        d = datetime.fromtimestamp(r[0]).strftime("%Y-%m-%d")
        if d in days:
            days[d] += 1
    return {
        "people": people, "peopleToday": people_today, "tryons": tries, "tryonsToday": tries_today,
        "successRate": round(ok / tries * 100) if tries else None,
        "avgSeconds": round(avg / 1000) if avg else None,
        "topWigs": wigs, "daily": [{"date": k, "count": v} for k, v in days.items()],
    }


@router.get("/sessions")
def sessions(q: str = "", limit: int = Query(30, le=200), offset: int = 0, token: str = Depends(require_admin)):
    like = f"%{q.strip()}%"
    with db.conn() as c:
        total = c.execute(
            "SELECT COUNT(*) FROM sessions WHERE name LIKE ? OR phone LIKE ? OR email LIKE ?", (like, like, like)
        ).fetchone()[0]
        rows = c.execute(
            """SELECT s.*, COUNT(t.id) AS tries,
                      SUM(CASE WHEN t.status = 'ok' THEN 1 ELSE 0 END) AS ok,
                      GROUP_CONCAT(DISTINCT t.wig_name) AS wigs,
                      MAX(t.created_at) AS last_try
               FROM sessions s LEFT JOIN tryons t ON t.session_id = s.id
               WHERE s.name LIKE ? OR s.phone LIKE ? OR s.email LIKE ?
               GROUP BY s.id ORDER BY s.created_at DESC LIMIT ? OFFSET ?""",
            (like, like, like, limit, offset),
        ).fetchall()
    items = []
    for r in rows:
        items.append({
            "id": r["id"], "name": r["name"], "phone": r["phone"], "email": r["email"],
            "createdAt": r["created_at"], "lastTry": r["last_try"], "tries": r["tries"], "ok": r["ok"] or 0,
            "wigs": [w for w in (r["wigs"] or "").split(",") if w],
            "selfie": media_url(r["selfie"], token), "device": short_device(r["user_agent"]),
        })
    return {"total": total, "items": items}


@router.get("/sessions/{sid}")
def session_detail(sid: str, token: str = Depends(require_admin)):
    with db.conn() as c:
        s = c.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if not s:
            raise HTTPException(404, "Not found")
        tries = c.execute("SELECT * FROM tryons WHERE session_id = ? ORDER BY created_at DESC", (sid,)).fetchall()
    return {
        "id": s["id"], "name": s["name"], "phone": s["phone"], "email": s["email"],
        "consent": bool(s["consent"]), "createdAt": s["created_at"], "ip": s["ip"],
        "device": short_device(s["user_agent"]), "userAgent": s["user_agent"],
        "selfie": media_url(s["selfie"], token),
        "tryons": [{
            "id": t["id"], "wigId": t["wig_id"], "wigName": t["wig_name"], "status": t["status"],
            "error": t["error"], "faceLocked": bool(t["face_locked"]), "seconds": round((t["duration_ms"] or 0) / 1000),
            "createdAt": t["created_at"], "before": media_url(t["before_img"], token),
            "after": media_url(t["after_img"], token),
        } for t in tries],
    }


@router.delete("/sessions/{sid}")
def remove(sid: str, _: str = Depends(require_admin)):
    if not db.delete_session(sid):
        raise HTTPException(404, "Not found")
    return {"ok": True}


@router.get("/export.csv")
def export(_: str = Depends(require_admin)):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["name", "phone", "email", "first_seen", "wig", "status", "tried_at", "seconds", "device"])
    with db.conn() as c:
        rows = c.execute(
            """SELECT s.name, s.phone, s.email, s.created_at, s.user_agent, t.wig_name, t.status,
                      t.created_at AS tried, t.duration_ms
               FROM sessions s LEFT JOIN tryons t ON t.session_id = s.id ORDER BY s.created_at DESC, t.created_at"""
        ).fetchall()
    fmt = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
    for r in rows:
        w.writerow([r["name"], r["phone"] or "", r["email"] or "", fmt(r["created_at"]), r["wig_name"] or "",
                    r["status"] or "", fmt(r["tried"]), round((r["duration_ms"] or 0) / 1000) or "",
                    short_device(r["user_agent"])])
    buf.seek(0)
    name = f"tryon-customers-{datetime.now():%Y%m%d}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


def short_device(ua: str | None) -> str:
    ua = ua or ""
    os_ = ("iPhone" if "iPhone" in ua else "iPad" if "iPad" in ua else "Android" if "Android" in ua
           else "Windows" if "Windows" in ua else "Mac" if "Mac OS" in ua else "Other")
    br = ("Edge" if "Edg/" in ua else "Chrome" if "Chrome" in ua else "Safari" if "Safari" in ua
          else "Firefox" if "Firefox" in ua else "")
    return f"{os_} · {br}" if br else os_


# ---------------- status ----------------

@router.get("/health")
def health(_: str = Depends(require_admin)):
    return {**STATUS, "openaiKey": bool(os.getenv("OPENAI_API_KEY"))}


# ---------------- wig catalogue ----------------

WIG_FIELDS = ("id", "name", "price", "image", "thumb", "description", "bangs", "buyUrl", "active", "updated")


def _slug(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "wig"
    return f"{base}-{uuid.uuid4().hex[:4]}"


def _save_wig_image(raw: bytes, wig_id: str) -> tuple[str, str, str | None]:
    """Store the product photo and a grid thumbnail. Returns (image, thumb, warning)."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    except Exception:
        raise HTTPException(400, "That file is not an image we can read")
    if img.mode in ("RGBA", "LA", "P"):  # transparent PNGs: put the wig on a plain light-grey background
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (233, 233, 233, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    img.thumbnail((1536, 1536))
    stamp = int(time.time())
    catalog_store.WIGS.mkdir(exist_ok=True)
    catalog_store.THUMBS.mkdir(exist_ok=True)
    name = f"{wig_id}-{stamp}"
    img.save(catalog_store.WIGS / f"{name}.png", optimize=True)
    t = img.copy()
    t.thumbnail((480, 720))
    t.save(catalog_store.THUMBS / f"{name}.webp", quality=82)

    from faces import wig_shape  # check the AI pipeline can read this photo
    warning = None
    if wig_shape(np.array(img)) is None:
        warning = ("We couldn't make out the wig against its background. For best results use a "
                   "front-facing photo of the wig alone on a plain, light background.")
    return f"wigs/{name}.png", f"wigs/thumbs/{name}.webp", warning


def _remove_files(p: dict):
    for key in ("image", "thumb"):
        if p.get(key):
            f = (catalog_store.ROOT / p[key]).resolve()
            if catalog_store.WIGS.resolve() in f.parents:
                f.unlink(missing_ok=True)


def _public(p: dict) -> dict:
    return {k: p.get(k) for k in WIG_FIELDS} | {"active": p.get("active", True)}


def _truthy(v: str | None) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


@router.get("/wigs")
def list_wigs(_: str = Depends(require_admin)):
    return {"items": [_public(p) for p in catalog_store.load()["products"]]}


@router.post("/wigs")
async def add_wig(
    image: UploadFile = File(...),
    name: str = Form(...),
    price: str = Form(""),
    description: str = Form(""),
    buyUrl: str = Form(""),
    bangs: str = Form("false"),
    _: str = Depends(require_admin),
):
    name = name.strip()[:80]
    if len(name) < 2:
        raise HTTPException(400, "Please give the wig a name")
    raw = await image.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image is too large (max 20 MB)")
    wig_id = _slug(name)
    img, thumb, warning = _save_wig_image(raw, wig_id)
    item = {"id": wig_id, "name": name, "price": price.strip()[:40], "image": img, "thumb": thumb,
            "description": description.strip()[:600], "bangs": _truthy(bangs), "buyUrl": buyUrl.strip()[:500],
            "active": True, "updated": int(time.time())}
    data = catalog_store.load()
    data["products"].insert(0, item)
    catalog_store.save(data)
    return {"item": _public(item), "warning": warning}


@router.patch("/wigs/{wig_id}")
async def edit_wig(
    wig_id: str,
    image: UploadFile | None = File(None),
    name: str | None = Form(None),
    price: str | None = Form(None),
    description: str | None = Form(None),
    buyUrl: str | None = Form(None),
    bangs: str | None = Form(None),
    active: str | None = Form(None),
    _: str = Depends(require_admin),
):
    data = catalog_store.load()
    p = next((x for x in data["products"] if x["id"] == wig_id), None)
    if not p:
        raise HTTPException(404, "Wig not found")
    warning = None
    if image is not None and image.filename:
        raw = await image.read()
        old = dict(p)
        p["image"], p["thumb"], warning = _save_wig_image(raw, wig_id)
        _remove_files(old)
        catalog_store.forget_shape(wig_id)
    if name is not None and name.strip():
        p["name"] = name.strip()[:80]
    if price is not None:
        p["price"] = price.strip()[:40]
    if description is not None:
        p["description"] = description.strip()[:600]
    if buyUrl is not None:
        p["buyUrl"] = buyUrl.strip()[:500]
    if bangs is not None:
        p["bangs"] = _truthy(bangs)
    if active is not None:
        p["active"] = _truthy(active)
    p["updated"] = int(time.time())
    catalog_store.save(data)
    return {"item": _public(p), "warning": warning}


@router.post("/wigs/{wig_id}/move")
def move_wig(wig_id: str, direction: str = Query(..., pattern="^(up|down)$"), _: str = Depends(require_admin)):
    data = catalog_store.load()
    ids = [p["id"] for p in data["products"]]
    if wig_id not in ids:
        raise HTTPException(404, "Wig not found")
    i = ids.index(wig_id)
    j = i - 1 if direction == "up" else i + 1
    if 0 <= j < len(ids):
        data["products"][i], data["products"][j] = data["products"][j], data["products"][i]
        catalog_store.save(data)
    return {"ok": True}


@router.delete("/wigs/{wig_id}")
def delete_wig(wig_id: str, _: str = Depends(require_admin)):
    data = catalog_store.load()
    p = next((x for x in data["products"] if x["id"] == wig_id), None)
    if not p:
        raise HTTPException(404, "Wig not found")
    data["products"] = [x for x in data["products"] if x["id"] != wig_id]
    catalog_store.save(data)
    _remove_files(p)
    catalog_store.forget_shape(wig_id)
    return {"ok": True}
