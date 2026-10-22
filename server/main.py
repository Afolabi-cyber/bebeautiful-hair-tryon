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
from faces import BODY_SKIN, CLOTHES, FACE_OVAL, beautify, clean_photo, HAIR, LEFT_BROW, LEFT_EYE, LIPS, NOSTRILS, RIGHT_BROW, RIGHT_EYE, FaceTool, align, geom, backdrop, clothes_keep, compose_ai, face_lock, keep_clothes, make_room, Segmenter, hair_zone, head_crop, keep_mask, keep_skin_tone, skin_tone_gap, old_hair, openai_mask_png, photo_problem, skin_lock, transfer_hair, wig_shape

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "high")  # low | medium | high
# Background of the result: studio (clean light backdrop), white, or original (the customer's own)
RESULT_BACKGROUND = os.getenv("RESULT_BACKGROUND", "studio").lower()
# "ai" (default): the AI renders the whole head (face, hair, light and shadows together, lightly
# beautified); the face is then locked back to the customer's exact shape and features and her clothes
# always come from her photo. "photo": her original face and photo with only the AI hair laid on top.
BLEND_MODE = os.getenv("BLEND_MODE", "pure").strip().lower()
# "pure" (default): the AI makes the final picture; the server only prepares the photo and checks the
# result (same person, same clothes), regenerating if a check fails. Nothing is pasted back in.
# 1 = the result keeps her exact skin tone, 0 = whatever the AI drew
# on = her real face (smoothed only) is used in the result, never the AI redraw of it
KEEP_FACE = os.getenv("KEEP_FACE", "on").lower() not in ("0", "off", "false", "no")
SKIN_TONE = max(0.0, min(1.0, float(os.getenv("SKIN_TONE", "1"))))
TRYON_ATTEMPTS = max(1, min(3, int(os.getenv("TRYON_ATTEMPTS", "3"))))
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
        "to the scalp, the hair falling naturally around the face and over the shoulders and covering both ears "
        "completely, the same on the left and the right (do not draw any ears), "
        "with the hair casting soft natural shadows where it touches the forehead, cheeks, neck and shoulders. "
        "Hide all of her own hair, scarf or cap completely under the wig. The light on the face, hair and "
        "clothes must match, as in one real photograph.\n"
        "Gently beautify her face the way a professional retoucher and makeup artist would, keeping it natural: "
        "even skin tone, smooth out blemishes, spots and dark circles, a soft healthy glow, neat brows, subtle "
        "natural lip colour. Keep real skin texture, never plastic or painted. She must stay unmistakably the "
        "same person: exactly the same face shape, eyes, eye shape and spacing, eyebrow shape, nose, lips, "
        "jawline, age and expression, and exactly the same skin tone (do not lighten or darken her skin). "
        "Do not add anything that is not in image 1: no glasses, earrings, jewellery or accessories she is not "
        "already wearing. If she wears glasses or a necklace, keep them exactly as they are.\n"
        "Keep her clothing exactly as in image 1: the same garment, colour, pattern, neckline and fit. Keep the "
        "same pose, head size, position and framing as image 1. Replace the background with a plain, seamless, "
        "softly lit light grey-white studio backdrop. Shot on a good camera, photorealistic, not an illustration."
    )


def build_prompt_pure(p: dict, retry_note: str = "", outfit: str = "") -> str:
    desc = p.get("description") or p["name"]
    return (
        "Image 1 is a photo of a customer. Image 2 is a product photo of a wig (shown on its own, face area empty).\n"
        f"Wig: {desc}.\n\n"
        "Create ONE finished, photorealistic portrait of the customer from image 1 wearing this exact wig, as if "
        "photographed in a professional studio. It must look like a real photograph, never like a collage or an edit.\n\n"
        "THE WIG: reproduce image 2 faithfully: the same braid size and pattern, parting, curl type and where the curls "
        "start, the same colours and ombre in the same places, the same length, density and fullness. Do not invent a "
        "different style. Fit it like a professionally installed wig: flush and snug on the scalp, the top close to the "
        "head, a soft natural hairline blending into her forehead and temples, no visible lace, cap or edges. The hair "
        "covers both ears completely and falls naturally over the shoulders and down the front and back, as long as the "
        "wig is, with real volume, shadows and light, casting soft shadows on the neck and shoulders where it touches "
        "them. Completely hide her own hair and anything on her head (scarf, cap, hood, headphones, bonnet) under the wig.\n\n"
        "THE FACE: the face in image 1 is already professionally retouched. Keep it EXACTLY as it is: do not redraw, "
        "reshape, resize, smooth, sharpen, slim, enlarge the eyes or change it in any way. The same face shape, eyes, "
        "eyebrows, nose, lips, jaw, chin, expression, skin texture and exactly the same skin tone and complexion. Do not lighten, brighten, whiten, darken or tint her skin, and do not airbrush it: the natural skin texture stays. Keep the head exactly "
        "where it is in image 1, at the same size and angle. Only the hairline at the top of the forehead and the hair "
        "around the face are new. Keep glasses only if she is wearing them; do not add glasses, earrings, jewellery or "
        "anything else she is not wearing.\n\n"
        "THE CLOTHES: keep her clothing exactly as in image 1: the same garment, colour, pattern, print, neckline, collar "
        "and fit, and keep any necklace. Never invent, replace or restyle her clothes, and never add a pattern or print. "
        + (f"She is wearing: {outfit}. It must stay exactly that. " if outfit else "")
        + "The wig's hair may fall in front of the clothes.\n\n"
        "THE PHOTO: soft, even studio light that is the same on the face, hair, neck and clothes and keeps her real skin tone. A plain, "
        "seamless, softly lit light grey-white studio backdrop, with nothing else in the background. Head-and-shoulders "
        "framing as in image 1. If any part of image 1 is blurred or empty at the edges, continue the hair, shoulders "
        "and backdrop naturally there. Sharp, clean, high quality, photorealistic."
        + (f"\n\nIMPORTANT, previous attempt failed: {retry_note}" if retry_note else "")
    )


def check_result(crop: np.ndarray, lm: np.ndarray, out: np.ndarray, out_lm, seg_src, seg_out) -> tuple[float, str]:
    """How good an AI result is (0 = perfect, higher = worse) and what went wrong, for a retry.
    Checks only; the picture itself is never edited."""
    if out_lm is None:
        return 99.0, "the face was not clearly visible; show her face clearly, looking at the camera"
    M = align(lm, out_lm, strict=False)
    if M is None:
        return 50.0, "the face moved, changed size or changed shape; keep her face exactly as in image 1"
    g = geom(lm)
    mapped = lm @ M[:, :2].T + M[:, 2]
    # identity geometry: how far her eyes, brows, nose, lips and face outline moved relative to each other
    ids = np.unique(np.concatenate([LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS, NOSTRILS, FACE_OVAL]))
    shape_err = float(np.percentile(np.linalg.norm(mapped[ids] - out_lm[ids], axis=1), 75)) / g.height
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    shift = float(np.linalg.norm(M[:, 2] + lm.mean(0) @ M[:, :2].T - lm.mean(0))) / g.height
    score, why = 0.0, []
    # calibrated on real faces: same person (retouched, relit) <= 0.016, a different person >= 0.049
    if shape_err > 0.022:
        score += (shape_err - 0.022) * 400 + 3
        why.append("her facial features changed; keep exactly the same eyes, nose, lips and face shape")
    if abs(scale - 1) > 0.1 or shift > 0.25:
        score += 10
        why.append("the head moved or changed size; keep the same framing as image 1")
    if seg_src is not None and seg_out is not None:
        H, W = crop.shape[:2]
        cl = (seg_src[..., CLOTHES] > 0.6) & ~cv2.dilate((old_hair(seg_src, lm) > 0.4).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        Minv = cv2.invertAffineTransform(M)
        out_s = cv2.warpAffine(out, Minv, (W, H), borderMode=cv2.BORDER_REPLICATE)
        hair_o = cv2.warpAffine(np.ascontiguousarray(seg_out[..., HAIR]), Minv, (W, H)) > 0.3
        cl &= ~cv2.dilate(hair_o.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        if cl.sum() > 2000:
            la = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
            lb = cv2.cvtColor(out_s, cv2.COLOR_RGB2LAB).astype(np.float32)
            # colour of the garment (broad): allow the lighting to change a little, not the colour
            dab = float(np.linalg.norm(la[cl][:, 1:].mean(0) - lb[cl][:, 1:].mean(0)))
            dL = abs(float(la[cl][:, 0].mean() - lb[cl][:, 0].mean()))
            # pattern: a plain garment must not come back printed (camouflage, florals), nor the reverse
            dtex = abs(float(lb[cl][:, 0].std() - la[cl][:, 0].std())) + 0.5 * abs(
                float(lb[cl][:, 1:].std(0).mean() - la[cl][:, 1:].std(0).mean()))
            if dtex > 10:
                score += (dtex - 10) * 0.5 + 6
                why.append("her clothes were changed or a pattern was added; keep exactly the same garment, "
                           "colour and pattern as in image 1")
            if dab > 9 or dL > 30:
                score += max(dab - 9, 0) + max(dL - 30, 0) * 0.5 + 5
                why.append("her clothes changed colour; keep her clothing exactly as in image 1")
        gap = skin_tone_gap(crop, seg_src, out, seg_out)
        if gap is not None:
            dL, dc = gap
            # lightness only: the colour of her photo is often off (office or bulb light), not her skin
            if abs(dL) > 7:
                score += (abs(dL) - 7) * 0.5 + 2
                how = "lighter" if dL > 0 else "darker"
                why.append(f"her skin was made {how}; keep exactly her own skin tone from image 1, "
                           "do not lighten, brighten or airbrush her skin")
    return score, "; ".join(why)


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
    if BLEND_MODE != "pure" and segmenter() is None and not ALLOW_WITHOUT_HAIR_MODEL:
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


OUTFIT_MODEL = os.getenv("OUTFIT_MODEL", "gpt-4o-mini")


async def describe_outfit(crop: np.ndarray) -> str:
    """One line describing exactly what she is wearing, so the image prompt can name it (the image
    model keeps clothes far better when told what they are). Empty if it can't be had quickly."""
    try:
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        res = await asyncio.wait_for(openai_client().chat.completions.create(
            model=OUTFIT_MODEL, max_tokens=60, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe only the clothing this person wears on the upper body, in one short "
                 "line: garment type, exact colour, plain or the exact pattern, neckline or collar, buttons, and any "
                 "necklace or glasses. No other words."},
                {"type": "image_url", "image_url": {"url": url, "detail": "low"}}]}]), timeout=15)
        text = (res.choices[0].message.content or "").strip().replace("\n", " ")
        return text[:240]
    except Exception as e:
        log.warning("outfit description skipped: %s", e)
        return ""


async def ai_edit(crop: np.ndarray, wig_rgb: np.ndarray, prompt: str, keep: np.ndarray | None = None) -> np.ndarray:
    kwargs = dict(
        model=MODEL,
        image=[("customer.png", encode(crop), "image/png"), ("wig.png", encode(wig_rgb), "image/png")],
        prompt=prompt,
        size=f"{OUT_W}x{OUT_H}",
        quality=QUALITY,
        output_format="png",
        n=1,
    )
    if keep is not None:
        kwargs["mask"] = ("mask.png", openai_mask_png(keep), "image/png")  # opaque = do not edit
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
    return out


async def run_pure(crop: np.ndarray, lm: np.ndarray, wig_rgb: np.ndarray, p: dict, seg_src, added,
                   src_scale: float = 1.0) -> dict:
    """The AI makes the final picture. The server only checks it (same person, same clothes, face in
    place) and asks the AI again with a note on what went wrong if a check fails; the best is kept."""
    # Skin smoothing on HER real face (tone, blemishes, noise; features and shape untouched), done on the
    # photo before the AI sees it. The AI is then told to keep that face exactly and builds everything
    # else around it, so the result is one natural photo with her own unchanged face.
    body = None
    if seg_src is not None:
        body = cv2.GaussianBlur(np.clip((seg_src[..., BODY_SKIN] - 0.3) / 0.4, 0, 1), (0, 0), max(2.0, 0.01 * OUT_W))
    sent = clean_photo(crop, lm, src_scale)
    if FACE_ENHANCE > 0:
        sent = beautify(sent, lm, FACE_ENHANCE, style="natural", skin_extra=body)
    # Protect the face (brows to chin) in the AI edit; the forehead/hairline stays free for the wig
    face_keep = keep_mask(crop.shape, lm, shrink=0.92)

    outfit = await describe_outfit(crop)
    if outfit:
        log.info("outfit: %s", outfit)
    best, note, tries = None, "", 0
    for attempt in range(TRYON_ATTEMPTS):
        tries += 1
        out = await ai_edit(sent, wig_rgb, build_prompt_pure(p, note, outfit), keep=face_keep)
        out_lm = await landmarks(out)
        seg_out = None
        if seg_src is not None:
            try:
                seg_out = await segment(out)
            except Exception:
                seg_out = None
        score, why = check_result(crop, lm, out, out_lm, seg_src, seg_out)
        log.info("try-on attempt %d: score %.1f %s", tries, score, why or "(passed)")
        if best is None or score < best[0]:
            best = (score, out, seg_out, out_lm, why)
        if score < 1.0:
            break
        note = why
    score, out, seg_out, out_lm, why = best
    # Last resort, only if every attempt still changed her clothes: put her real clothes back where
    # the new hair isn't covering them. Her clothes are never allowed to be made up.
    if "clothes" in why and seg_src is not None and seg_out is not None:
        M = align(lm, out_lm, strict=False) if out_lm is not None else None
        out, frac = keep_clothes(crop, out, M, seg_src, seg_out)
        log.info("clothes restored from her photo (%.0f%% of the picture)", frac * 100)
    # Her face stays her face: her own (smoothed) face goes back on the result, and the result's skin
    # (neck, ears, chest) is first brought to her exact skin colour so there is no edge where they meet.
    locked = False
    if SKIN_TONE > 0 and seg_src is not None and seg_out is not None:
        gap = skin_tone_gap(sent, seg_src, out, seg_out)
        out = keep_skin_tone(sent, seg_src, out, seg_out, SKIN_TONE, exact=KEEP_FACE)
        log.info("skin tone kept (was off by L %.1f, colour %.1f)", *(gap or (0, 0)))
    if KEEP_FACE:
        out, locked = face_lock(sent, lm, out, out_lm, bool(p.get("bangs")), enhance=0, relight=0.4)
        log.info("her own face kept: %s", locked)
    before = crop
    if added is not None and added.any():
        a_ = cv2.GaussianBlur(added.astype(np.float32), (0, 0), 3)[..., None]
        before = np.clip(crop * (1 - a_) + backdrop(OUT_H, OUT_W, "studio") * a_, 0, 255).astype(np.uint8)
    mode = f"pure ai ({tries} attempt{'s' if tries > 1 else ''}, check score {score:.1f})"
    log.info("try-on finished: mode=%s", mode)
    return {"before": before, "after": out, "locked": True, "mode": mode}


async def run_tryon(rgb: np.ndarray, p: dict) -> dict:
    # 1. Find the face and crop a head-and-shoulders portrait at the output aspect ratio.
    lm = await landmarks(rgb)
    if lm is None:
        raise HTTPException(422, "We couldn't find a face in your photo. Please retake it.")
    # Close selfies: extend the canvas so there's room for the full wig above the head and the shoulders
    rgb, lm, valid = make_room(rgb, lm, OUT_W / OUT_H)
    x, y, w, h = head_crop(rgb.shape, lm, OUT_W / OUT_H)
    # INTER_AREA only for shrinking; enlarging with it duplicates pixels into visible blocks
    crop = cv2.resize(rgb[y:y + h, x:x + w], (OUT_W, OUT_H),
                      interpolation=cv2.INTER_AREA if w >= OUT_W else cv2.INTER_LANCZOS4)
    added = cv2.resize(valid[y:y + h, x:x + w], (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST) == 0
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
        if seg_src is not None and added.any():
            seg_src[added] = 0
            seg_src[added, 0] = 1  # the added area is not part of her photo: nothing of hers is there
    except Exception:
        log.exception("Hair model failed on this photo")
        seg_src = None
    if seg_src is not None:
        own = (old_hair(seg_src, lm) > 0.35).astype(np.uint8)  # her hair, or a wrap/cap/wig she wears
        g = max(3, int(0.03 * OUT_W))
        own = cv2.dilate(own, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        zone = np.maximum(zone, own)
        zone[skin_lock(seg_src, lm, bangs) > 0] = 0  # her real forehead/temples: the wig is built around them
    if BLEND_MODE == "pure":
        return await run_pure(crop, lm, wig_rgb, p, seg_src, added, src_scale=w / OUT_W)

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
            out_final = compose_ai(crop, lm, out, out_lm, seg_src, seg_out, RESULT_BACKGROUND, src_scale=w / OUT_W,
                                   enhance=FACE_ENHANCE, added=added)
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
    before = crop
    if added.any():  # the "before" picture shows the clean backdrop where the canvas was extended
        a_ = cv2.GaussianBlur(added.astype(np.float32), (0, 0), 3)[..., None]
        before = np.clip(crop * (1 - a_) + backdrop(OUT_H, OUT_W, "studio") * a_, 0, 255).astype(np.uint8)
    return {"before": before, "after": out, "locked": locked, "mode": mode}


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
