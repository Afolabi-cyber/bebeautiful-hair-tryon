"""Hair Try-On API: selfie + chosen wig -> OpenAI image edit -> new hair placed on the ORIGINAL photo."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIStatusError, AsyncOpenAI
from PIL import Image, ImageOps

import catalog_store
import db
from status import STATUS
from admin import router as admin_router
from faces import HAIR, FaceTool, clothes_keep, compose_ai, Segmenter, hair_zone, head_crop, keep_mask, old_hair, openai_mask_png, photo_problem, skin_lock, transfer_hair, wig_shape

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "high")  # low | medium | high
# Background of the result: studio (clean light backdrop), white, or original (the customer's own)
RESULT_BACKGROUND = os.getenv("RESULT_BACKGROUND", "studio").lower()
# "ai" (default): the AI renders the whole head (face, hair, light and shadows together, lightly
# beautified); the face is then locked back to the customer's exact shape and features and her clothes
# always come from her photo. "photo": her original face and photo with only the AI hair laid on top.
BLEND_MODE = os.getenv("BLEND_MODE", "ai").strip().lower()
FACE_ENHANCE = min(1.0, max(0.0, float(os.getenv("FACE_ENHANCE", "0.75"))))  # retouch strength: 0 = off, 0.75 = default, 1 = strongest
OUT_W, OUT_H = 1024, 1536
MAX_UPLOAD = 15 * 1024 * 1024
MAX_TRIES = int(os.getenv("MAX_TRYONS_PER_PERSON", "10"))  # cost guard per customer session
# Public-facing limits per device/IP per hour (stops abuse and runaway OpenAI costs)
SESSIONS_PER_HOUR = int(os.getenv("PHOTOS_PER_IP_PER_HOUR", "15"))
TRYONS_PER_HOUR = int(os.getenv("TRYONS_PER_IP_PER_HOUR", "30"))
# Without the hair model the clothes can't be fully protected, so try-ons are paused by default.
ALLOW_WITHOUT_HAIR_MODEL = os.getenv("ALLOW_WITHOUT_HAIR_MODEL", "off").lower() == "on"

log = logging.getLogger("tryon")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Hair Try-On")
app.include_router(admin_router)
db.init()
client: AsyncOpenAI | None = None
_faces: FaceTool | None = None
_segmenter: Segmenter | None = None
_seg_failed_at = 0.0  # last failed load; retried every 2 minutes
_cache: OrderedDict[str, dict] = OrderedDict()
_face_lock_guard = asyncio.Lock()  # MediaPipe detector is not thread safe


def faces() -> FaceTool:
    global _faces
    if _faces is None:
        _faces = FaceTool()
    return _faces


def segmenter() -> Segmenter | None:
    global _segmenter, _seg_failed_at
    if _segmenter is None and time.time() - _seg_failed_at > 120:
        try:
            _segmenter = Segmenter()
            STATUS.update(hair="ok", hairError=None)
            log.info("MODEL CHECK  hair/clothes model ........ OK")
        except Exception as e:  # e.g. the model download is blocked on this network
            _seg_failed_at = time.time()
            STATUS.update(hair="failed", hairError=f"{type(e).__name__}: {e}")
            log.error("MODEL CHECK  hair/clothes model ........ FAILED: %s\n"
                      "  Try-ons are paused until it loads (clothes could change without it).\n"
                      "  Fix: download %s\n  and save it as %s", STATUS["hairError"],
                      __import__("faces").SEG_URL, __import__("faces").SEG_PATH)
    return _segmenter


@app.on_event("startup")
async def check_models():
    """Load the models at startup and say clearly in the console whether they work."""
    try:
        await asyncio.to_thread(faces)
        STATUS["face"] = "ok"
        log.info("MODEL CHECK  face landmarks ............ OK")
    except Exception as e:
        STATUS["face"] = "failed"
        log.error("MODEL CHECK  face landmarks ............ FAILED: %s", e)
    await asyncio.to_thread(segmenter)


async def segment(rgb: np.ndarray) -> np.ndarray | None:
    seg = segmenter()
    if seg is None:
        return None
    async with _face_lock_guard:
        return await asyncio.to_thread(seg.probs, rgb)


def openai_client() -> AsyncOpenAI:
    global client
    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(500, "Server is missing OPENAI_API_KEY (see .env.example)")
        client = AsyncOpenAI(timeout=240)
    return client


def catalog() -> dict:
    return catalog_store.load()


def product(wig_id: str) -> dict:
    for p in catalog()["products"]:
        if p["id"] == wig_id and p.get("active", True):
            return p
    raise HTTPException(404, "Unknown wig")


def supports_fidelity(model: str) -> bool:
    # input_fidelity="high" exists on gpt-image-1 and gpt-image-1.5 (not mini). gpt-image-2+ must omit it.
    return model.startswith(("gpt-image-1.5", "gpt-image-1")) and "mini" not in model


def build_prompt(p: dict) -> str:
    desc = p.get("description") or p["name"]
    return (
        "Image 1 is a photo of a customer. Image 2 is a product photo of a wig on a plain background, "
        "shown from the front with the face area empty.\n"
        f"Wig: {desc}.\n"
        "Edit image 1 so the customer is wearing this exact wig. Reproduce the wig from image 2 "
        "faithfully: the same braid size and pattern, the same parting, the same curl type and where "
        "the curls start, the same colours and ombre in the same places, the same length and volume. "
        "Do not invent a different hairstyle and do not change the wig's colours. Seat the wig "
        "flush and snug on the customer's head like a professionally installed wig: it hugs the scalp "
        "and the shape of the skull at the crown and sides, with no gap, lift or space between the wig "
        "and the head, the top of the wig lying close to the scalp (not puffed up or floating above the head), no visible cap, lace or edges, and the wig's hairline blends naturally into the "
        "skin of their forehead and temples. Hide all of their own hair underneath it so none of it "
        "shows at the hairline, temples, sides or neck. The hair hangs down in front of both ears and covers them, the same on the left and the right side, so neither ear shows. Let the hair fall over the shoulders and down the body as long as the wig "
        "is, and match the camera angle of image 1. Use soft, even, flattering light on the face and hair, "
        "like a good beauty photo, so it looks like a real professional photo.\n"
        "Do not change the customer in any other way. Keep their face identical: same facial features, "
        "face shape, skin tone and texture, eyes, eyebrows, nose, lips, makeup and expression. No beautifying, slimming, smoothing, lightening or retouching.\n"
        "Keep the customer's clothing exactly as in image 1: the same garment, colour, pattern, neckline "
        "and fit. Do not darken, recolour, replace or cover the clothes with anything except the wig's "
        "hair where it naturally falls in front of the shoulders and chest. Keep the body, pose, "
        "background and framing exactly as in image 1. Photorealistic."
    )


def build_prompt_ai(p: dict) -> str:
    desc = p.get("description") or p["name"]
    return (
        "Image 1 is a photo of a customer. Image 2 is a product photo of a wig on a plain background, "
        "shown from the front with the face area empty.\n"
        f"Wig: {desc}.\n"
        "Make a beautiful, natural, photorealistic portrait of the customer from image 1 wearing this exact wig. "
        "Reproduce the wig from image 2 faithfully: the same braid size and pattern, the same parting, the same "
        "curl type and where the curls start, the same colours and ombre in the same places, the same length, "
        "density and volume. Do not invent a different hairstyle.\n"
        "The wig must look like it is really on her head: installed flush and snug on the scalp with a natural, "
        "soft hairline that blends into her forehead and temples, no visible lace, cap or edges, the top close "
        "to the scalp, the hair falling naturally around the face, in front of both ears and over the shoulders, "
        "with the hair casting soft natural shadows where it touches the forehead, cheeks, neck and shoulders. "
        "Hide all of her own hair, scarf or cap completely under the wig. The light on the face, hair and "
        "clothes must match, as in one real photograph.\n"
        "Gently beautify her face the way a professional retoucher and makeup artist would, keeping it natural: "
        "even skin tone, smooth out blemishes, spots and dark circles, a soft healthy glow, neat brows, subtle "
        "natural lip colour. Keep real skin texture, never plastic or painted. She must stay unmistakably the "
        "same person: exactly the same face shape, eyes, eye shape and spacing, eyebrow shape, nose, lips, "
        "jawline, age and expression, and exactly the same skin tone (do not lighten or darken her skin). "
        "Keep glasses, piercings and jewellery exactly as they are.\n"
        "Keep her clothing exactly as in image 1: the same garment, colour, pattern, neckline and fit. Keep the "
        "same pose, head size, position and framing as image 1. Replace the background with a plain, seamless, "
        "softly lit light grey-white studio backdrop. Shot on a good camera, photorealistic, not an illustration."
    )


def to_rgb(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "That file is not an image we can read")
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.array(img.convert("RGB"))


def encode(rgb: np.ndarray, fmt: str = ".png", quality: int = 92) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
    assert ok
    return buf.tobytes()


def data_url(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode()


async def landmarks(rgb: np.ndarray):
    async with _face_lock_guard:
        return await asyncio.to_thread(faces().landmarks, rgb)


@app.get("/api/health")
def public_health():
    """For Render's health check: up as long as the face model works."""
    if STATUS["face"] == "failed":
        raise HTTPException(503, "face model not loaded")
    return {"ok": True, "hairModel": STATUS["hair"]}


@app.get("/api/catalog")
def get_catalog():
    c = catalog()
    return {
        "brand": c.get("brand", {}),
        "products": [
            {k: p.get(k) for k in ("id", "name", "price", "image", "thumb", "buyUrl")}
            for p in c["products"] if p.get("active", True)
        ],
    }


_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(kind: str, ip: str, limit: int, window: int = 3600):
    q = _hits[f"{kind}:{ip}"]
    now = time.time()
    while q and q[0] < now - window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Lots of try-ons from this device. Please come back in a little while.")
    q.append(now)


def client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "")


@app.post("/api/sessions")
async def create_session(
    req: Request,
    selfie: UploadFile = File(...),
    name: str = Form(...),
    phone: str = Form(""),
    email: str = Form(""),
    consent: str = Form("false"),
):
    """Customer details + the photo taken before any wig is applied."""
    rate_limit("photo", client_ip(req), SESSIONS_PER_HOUR)
    name, phone, email = name.strip()[:80], phone.strip()[:30], email.strip()[:120]
    if len(name) < 2:
        raise HTTPException(400, "Please enter your name")
    if consent.lower() not in ("true", "1", "yes", "on"):
        raise HTTPException(400, "Please accept the photo consent to continue")
    raw = await selfie.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Photo is too large (max 15 MB)")

    rgb = to_rgb(raw)
    k = min(1.0, 1800 / max(rgb.shape[:2]))
    if k < 1:
        rgb = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    async with _face_lock_guard:
        lm0, blink = await asyncio.to_thread(faces().detect, rgb)
    if lm0 is None:
        raise HTTPException(422, "We couldn't find a face. Face the camera in good light and try again.")
    problem = photo_problem(rgb.shape, lm0, blink)
    if problem:
        raise HTTPException(422, problem)

    photo = db.save_photo(encode(rgb, ".jpg", 94))
    sid = db.create_session(name, phone or None, email or None, True, photo,
                            req.headers.get("user-agent", "")[:300], client_ip(req))
    return {"sessionId": sid}


@app.post("/api/tryon")
async def tryon(req: Request, session_id: str = Form(...), wig_id: str = Form(...)):
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session expired, please take your photo again")
    p = product(wig_id)

    key = f"{session_id}:{wig_id}:{p.get('updated', '')}:{MODEL}"
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    if segmenter() is None and not ALLOW_WITHOUT_HAIR_MODEL:
        raise HTTPException(503, "Virtual try-on is being set up. Please try again in a few minutes.")
    rate_limit("tryon", client_ip(req), TRYONS_PER_HOUR)
    if db.count_tryons(session_id, "ok") >= MAX_TRIES:
        raise HTTPException(429, f"You've reached the limit of {MAX_TRIES} try-ons. Thank you for trying!")

    started = time.time()
    try:
        result = await run_tryon(to_rgb((db.PHOTOS / s["selfie"]).read_bytes()), p)
    except HTTPException as e:
        db.add_tryon(session_id=session_id, wig_id=wig_id, wig_name=p["name"], status="error",
                     error=str(e.detail), model=MODEL, duration_ms=int((time.time() - started) * 1000))
        raise

    before_jpg, after_jpg = encode(result["before"], ".jpg"), encode(result["after"], ".jpg")
    db.add_tryon(session_id=session_id, wig_id=wig_id, wig_name=p["name"], status="ok",
                 before_img=db.save_photo(before_jpg), after_img=db.save_photo(after_jpg),
                 face_locked=int(result["locked"]), model=MODEL,
                 duration_ms=int((time.time() - started) * 1000))
    out = {"wig_id": wig_id, "before": data_url(before_jpg), "after": data_url(after_jpg),
           "faceLocked": result["locked"], "mode": result["mode"]}
    _cache[key] = out
    if len(_cache) > 50:
        _cache.popitem(last=False)
    return out


async def run_tryon(rgb: np.ndarray, p: dict) -> dict:
    # 1. Find the face and crop a head-and-shoulders portrait at the output aspect ratio.
    lm = await landmarks(rgb)
    if lm is None:
        raise HTTPException(422, "We couldn't find a face in your photo. Please retake it.")
    x, y, w, h = head_crop(rgb.shape, lm, OUT_W / OUT_H)
    crop = cv2.resize(rgb[y:y + h, x:x + w], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    lm = (lm - [x, y]) * [OUT_W / w, OUT_H / h]
    bangs = bool(p.get("bangs"))

    # 2. Ask OpenAI to put the wig on, with the face protected by a mask.
    wig_path = catalog_store.ROOT / p["image"]
    if not wig_path.exists():
        raise HTTPException(500, f"Wig image missing: {p['image']}")
    wig_rgb = to_rgb(wig_path.read_bytes())
    if p["id"] not in catalog_store.shapes:
        catalog_store.shapes[p["id"]] = wig_shape(wig_rgb)
    # The AI may only edit this fenced area (wig outline on the head + margin, minus the face).
    zone = hair_zone(crop.shape, lm, catalog_store.shapes[p["id"]], bangs)
    # Also let the AI repaint all of the customer's own hair, so it can hide it under the wig and
    # seat the wig flush instead of leaving old hair (or a gap) next to it. The face stays locked.
    try:
        seg_src = await segment(crop)
    except Exception:
        log.exception("Hair model failed on this photo")
        seg_src = None
    if seg_src is not None:
        own = (old_hair(seg_src, lm) > 0.35).astype(np.uint8)  # her hair, or a wrap/cap/wig she wears
        g = max(3, int(0.03 * OUT_W))
        own = cv2.dilate(own, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        zone = np.maximum(zone, own)
        zone[skin_lock(seg_src, lm, bangs) > 0] = 0  # her real forehead/temples: the wig is built around them
    ai_mode = BLEND_MODE == "ai" and seg_src is not None
    if ai_mode:
        # The AI may repaint everything (face, hair, background) except her clothes.
        cl = (clothes_keep(seg_src, lm) > 0.5).astype(np.uint8)
        e = max(2, int(0.01 * OUT_W))
        cl = cv2.erode(cl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
        edit_mask, prompt = (cl * 255).astype(np.uint8), build_prompt_ai(p)
    else:
        edit_mask, prompt = (zone == 0).astype(np.uint8) * 255, build_prompt(p)
    kwargs = dict(
        model=MODEL,
        image=[("customer.png", encode(crop), "image/png"), ("wig.png", encode(wig_rgb), "image/png")],
        mask=("mask.png", openai_mask_png(edit_mask), "image/png"),
        prompt=prompt,
        size=f"{OUT_W}x{OUT_H}",
        quality=QUALITY,
        output_format="png",
        n=1,
    )
    if supports_fidelity(MODEL):
        kwargs["input_fidelity"] = "high"
    try:
        res = await openai_client().images.edit(**kwargs)
    except APIStatusError as e:
        log.warning("OpenAI error %s: %s", e.status_code, e.message)
        msg = ("The AI couldn't process this photo. Try a different one."
               if e.status_code == 400 else "The AI is busy right now, please try again.")
        raise HTTPException(502, msg)
    out = to_rgb(base64.b64decode(res.data[0].b64_json))
    if out.shape[:2] != (OUT_H, OUT_W):
        out = cv2.resize(out, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)

    out_lm = await landmarks(out)

    # 3. The result is ALWAYS the customer's own photo. Outside the fenced zone nothing from the AI
    #    is used at all (no model needed for that). Inside it, the hair model (when available)
    #    keeps only the new hair, so the clothes under and between the braids are theirs too.
    seg_out = None
    try:
        if seg_src is not None:
            seg_out = await segment(out)
        if seg_out is not None and float(seg_out[..., HAIR].max()) < 0.5:
            seg_src = seg_out = None  # the model found no hair in the AI image
    except Exception:
        log.exception("Hair model failed on this image")
        seg_src = seg_out = None
    if seg_out is None and not ALLOW_WITHOUT_HAIR_MODEL:
        raise HTTPException(502, "We couldn't finish this look. Please try again.")
    out_final, mode = None, ""
    try:
        if ai_mode and seg_out is not None:
            out_final = compose_ai(crop, lm, out, out_lm, seg_src, seg_out, RESULT_BACKGROUND, src_scale=w / OUT_W)
            mode = "ai blend"
        if out_final is None:
            # AI face couldn't be matched to her photo reliably: use her own face with the AI hair
            out_final = transfer_hair(crop, lm, out, out_lm, zone, seg_src, seg_out, FACE_ENHANCE, RESULT_BACKGROUND,
                                      src_scale=w / OUT_W)
            mode = ("photo blend (fallback)" if ai_mode else "fence + hair model") if seg_out is not None else "fence only"
    except Exception:
        log.exception("Final composite failed")
        # Never fall back to the raw AI image (it may have changed the clothes).
        raise HTTPException(502, "We couldn't finish this look. Please try again.")
    out, locked = out_final, True
    log.info("try-on finished: mode=%s", mode)
    return {"before": crop, "after": out, "locked": locked, "mode": mode}


app.mount("/wigs", StaticFiles(directory=catalog_store.WIGS), name="wigs")

# The React app (frontend/dist, built with `npm run build`). Unknown paths fall back to index.html.
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
