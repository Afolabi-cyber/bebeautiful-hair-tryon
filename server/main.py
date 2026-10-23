"""Hair Try-On API: selfie + chosen wig -> OpenAI image edit -> new hair placed on the ORIGINAL photo."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
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
from faces import BODY_SKIN, CLOTHES, FACE_OVAL, beautify, clean_photo, HAIR, LEFT_BROW, LEFT_EYE, LIPS, NOSTRILS, RIGHT_BROW, RIGHT_EYE, FaceTool, align, geom, backdrop, clothes_keep, compose_ai, face_lock, garment_fit, added_hands, bare_scalp, face_match, head_gaps, neck_band, keep_body, match_backdrop, no_new_hands, tidy_body, keep_clothes, restore_clothes, make_room, Segmenter, hair_zone, head_crop, keep_mask, keep_skin_tone, skin_tone_gap, old_hair, openai_mask_png, photo_problem, skin_lock, transfer_hair, wig_shape

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "medium")  # low | medium | high
# Colour of the plain studio backdrop on every result (Be Beautiful Hair pink by default)
BACKDROP_HEX = os.getenv("BACKDROP_COLOR", "#f8b0d5").strip().lstrip("#")  # soft light tint of brand pink #ED3995
BACKDROP_RGB = tuple(int(BACKDROP_HEX[i:i + 2], 16) for i in (0, 2, 4))
BACKDROP_WORDS = f"soft light pink (exactly #{BACKDROP_HEX.upper()}, RGB {BACKDROP_RGB[0]}, {BACKDROP_RGB[1]}, {BACKDROP_RGB[2]})"
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
# on = a result whose clothes could not be kept is never shown (she is asked for a clearer photo)
# on = one extra AI retouch so the wig, her body and the backdrop blend into one photo (adds one AI call)
BLEND_PASS = os.getenv("BLEND_PASS", "off").lower() not in ("0", "off", "false", "no")  # off: the AI never redraws her body
# the blend pass only runs when the AI's body/clothes matched hers less than this (0..1); it runs at this quality
BLEND_BELOW = float(os.getenv("BLEND_BELOW", "0.75"))
BLEND_QUALITY = os.getenv("BLEND_QUALITY", "medium")
CLOTHES_STRICT = os.getenv("CLOTHES_STRICT", "off").lower() not in ("0", "off", "false", "no")
KEEP_CLOTHES = os.getenv("KEEP_CLOTHES", "on").lower() not in ("0", "off", "false", "no")
# "ai": the AI does a very light skin retouch and its face is used, but only if the identity check passed
# (same features, same face shape); otherwise her real face goes back. "photo": always her real face.
# on: a result that still shows added hands, a bare part of the head or bare scalp is never shown;
# the customer is asked for a new photo instead (protects the brand)
QUALITY_GATE = os.getenv("QUALITY_GATE", "off").lower() not in ("0", "off", "false", "no")
FACE_SOURCE = os.getenv("FACE_SOURCE", "ai").strip().lower()
# how closely the AI face's eyes, brows, nose and mouth must look like hers (0..1) to be used
FACE_MATCH_MIN = float(os.getenv("FACE_MATCH_MIN", "0.55"))
# light smoothing of her real face when it is used instead of the AI's (FACE_ENHANCE 0)
FALLBACK_SMOOTH = float(os.getenv("FALLBACK_SMOOTH", "0.6"))
KEEP_FACE = os.getenv("KEEP_FACE", "on").lower() not in ("0", "off", "false", "no")
SKIN_TONE = max(0.0, min(1.0, float(os.getenv("SKIN_TONE", "1"))))
TRYON_ATTEMPTS = max(1, min(3, int(os.getenv("TRYON_ATTEMPTS", "1"))))
FACE_ENHANCE = min(1.0, max(0.0, float(os.getenv("FACE_ENHANCE", "0"))))  # retouch strength: 0 = off, 0.75 = default, 1 = strongest
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
        "EDIT image 1. Change only two things: her hair becomes this exact wig, and the background becomes a plain "
        "studio backdrop. Everything else in image 1 stays exactly as it is, as if the same photo had been taken again: "
        "her face, her body, her pose, her hands and arms, her clothes and jewellery, their position, size and "
        "lighting. Do not re-imagine, re-pose or re-shoot her. The result must look like a real photograph, never like "
        "a collage.\n\n"
        "THE PARTING: the hair is parted in the middle of the head: a THIN, natural centre part line from the hairline "
        "to the crown, only a hair's width of scalp showing, with the braids starting right at the part on each side and "
        "the hair falling evenly to both sides (even if image 2 shows a side part). Never a wide bare strip, never bare "
        "scalp or forehead skin above the hairline: the whole top of the head is covered by the wig up to a natural, "
        "normal-height hairline (cover a high or receding hairline).\n\n"
        "THE WIG: reproduce image 2 faithfully: the same braid size and pattern, curl type and where the curls "
        "start, the same colours and ombre in the same places, the same length, density and fullness. Do not invent a "
        "different style. Fit it like a professionally installed wig: flush and snug on the scalp, the top close to the "
        "head, a soft natural hairline blending into her forehead and temples, no visible lace, cap or edges. The hair "
        "covers her WHOLE head: the crown, both temples and both sides, with no bare scalp, gap or background "
        "showing anywhere around her head (with a side part, braids still lie flat over the scalp on the other side). "
        "It covers both ears completely and falls evenly over BOTH shoulders, framing her face on both sides (some hair in front of each shoulder, even with a side part), naturally over the shoulders and down the front and back, as long as the "
        "wig is, full and dense with no thin or see-through patches, with real volume, shadows and light, casting soft shadows on the neck and shoulders where it touches "
        "them. Completely hide her own hair and anything on her head (scarf, cap, hood, headphones, bonnet) under the wig.\n\n"
        + ("THE FACE: give her skin a VERY LIGHT, natural retouch only: gently soften blemishes, spots and uneven "
           "texture and reduce oily shine, keeping her real pores and skin texture (it must still look like real skin, "
           "never airbrushed or plastic). Change nothing else: do not reshape, resize, slim or sharpen anything, do "
           "not enlarge the eyes; the same face shape, eyes, eyebrows, nose, lips, jaw, chin and expression, and "
           "exactly the same skin tone and complexion. Do not lighten, brighten, whiten, darken or tint her skin. "
           if FACE_SOURCE == "ai" else
           "THE FACE: the face in image 1 is already professionally retouched. Keep it EXACTLY as it is: do not redraw, "
           "reshape, resize, smooth, sharpen, slim, enlarge the eyes or change it in any way. The same face shape, eyes, "
           "eyebrows, nose, lips, jaw, chin, expression, skin texture and exactly the same skin tone and complexion. "
           "Do not lighten, brighten, whiten, darken or tint her skin, and do not airbrush it. ")
        + "Keep the head exactly "
        "where it is in image 1, at the same size and angle. Only the hairline at the top of the forehead and the hair "
        "around the face are new. Keep glasses only if she is wearing them; do not add glasses, earrings, jewellery or "
        "anything else she is not wearing.\n\n"
        "THE CLOTHES: keep her clothing exactly as in image 1: the same garment, colour, pattern, print, neckline, collar "
        "and fit, and keep any necklace. Never invent, replace or restyle her clothes, and never add a pattern or print. "
        + (f"She is wearing: {outfit}. It must stay exactly that. " if outfit else "")
        + "The wig's hair may fall in front of the clothes.\n\n"
        "THE PHOTO: keep the light on her as it is in image 1 and light the new hair to match it. A plain, "
        "seamless, evenly lit solid " + BACKDROP_WORDS + " studio backdrop that fills the whole frame edge to edge, with nothing of her original room (walls, ceiling, lights, furniture) left anywhere. Exactly the "
        "same framing and pose as image 1 (if her hands are near her face or she leans on something, keep that). If any part of image 1 is blurred or empty at the edges, fill it only with the hair and "
        "the backdrop. NO NEW HANDS OR ARMS: if a hand or arm is not visible in image 1 it must not appear in the "
        "result at all (no folded arms, no hands on the chin, no hands at the bottom of the picture). "
        "Never extend or invent her body or clothes: no shoulders, arms, sleeves or chest that are not "
        "visible in image 1, and the same neckline and collar. Sharp, clean, high quality, photorealistic."
        + (f"\n\nIMPORTANT, previous attempt failed: {retry_note}" if retry_note else "")
    )


def build_prompt_blend(p: dict, outfit: str = "") -> str:
    desc = p.get("description") or p["name"]
    return (
        "Image 1 is a try-on photo of a customer wearing a new wig. It was put together from parts (her real photo "
        "and the new hair), so the parts may not sit together naturally yet. Image 2 is the wig on its own for "
        f"reference. Wig: {desc}.\n\n"
        "Retouch image 1 into ONE seamless, natural photograph. Keep everything exactly where and as it is: her face, "
        "the wig (same style, braids, curls, colours, length and position), her clothes (the same garment, colour, "
        "pattern and fabric), her jewellery, her pose, arms and hands. Only fix how the parts meet:\n"
        "- the wig sits naturally on her head and falls naturally over her shoulders, arms and clothes, with soft, "
        "natural strand edges and soft contact shadows where hair touches her and her clothes;\n"
        "- her hands and arms stay exactly where and how they are in image 1 (same position, same fingers, same "
        "gesture): do not move, add, remove or re-pose any hand or arm. Never draw a hand that is not visible in "
        "image 1, and never draw the same hands twice; hands hidden behind the hair stay hidden;\n"
        "- where hair and hands or arms overlap, it looks physically right;\n"
        "- the outline of her body and clothes against the backdrop is clean and natural: no cut-out edges, halos, "
        "white or grey patches, holes or leftover pieces; any hole or gap in her clothes is filled with the same fabric;\n"
        "- do not add any arms, hands, table or surface: at the bottom of the frame her clothes simply continue "
        "down out of the picture;\n"
        "- one consistent light on her, the hair and the backdrop, keeping her real skin tone.\n"
        + (f"She is wearing: {outfit}. Her clothes must stay exactly that. " if outfit else "")
        + "Plain, seamless, evenly lit solid " + BACKDROP_WORDS + " studio backdrop, the same colour everywhere. "
        "Do not add or change anything else. Photorealistic."
    )


def is_long(p: dict) -> bool:
    """Shoulder-length or longer (the hair must fall on both shoulders); short styles are not held to that."""
    t = f"{p.get('name', '')} {p.get('description', '')}".lower()
    return not any(w in t for w in ("bob", "pixie", "short", "chin length", "chin-length", "tapered", "afro puff"))


def check_result(crop: np.ndarray, lm: np.ndarray, out: np.ndarray, out_lm, seg_src, seg_out,
                 long_hair: bool = True) -> tuple[float, str]:
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
    if seg_out is not None:
        # the wig must cover her whole head: hair all around the top and both sides of the face
        if bare_scalp(out_lm, seg_out):
            score += 12
            why.append("bare scalp or forehead skin shows on top of her head (a wide part or an uncovered hairline); "
                       "use a thin natural part and cover the whole top of the head down to a normal hairline")
        miss = head_gaps(lm @ M[:, :2].T + M[:, 2], seg_out, shoulders=long_hair)
        if miss:
            score += 12
            why.append(f"the wig does not cover the {miss} of her head (bare scalp or background shows there); the wig "
                       "must cover her whole head and both sides of the face like a real installed wig, even with a "
                       "side part: braids lie flat over the scalp on both sides and cover both ears")
    if seg_src is not None and seg_out is not None:
        hands = added_hands(seg_src, seg_out, M)
        if hands.any():
            score += 15
            why.append("hands or arms were added that are not in image 1; show no hands or arms at all unless "
                       "they are visible in image 1")
    if abs(scale - 1) > 0.06 or shift > 0.12:
        score += 10 + abs(scale - 1) * 50
        why.append("she was made smaller or bigger or moved; keep her exactly the same size and in the same place as "
                   "in image 1, same framing, do not zoom out and do not show more of her body than image 1 shows")
    if seg_src is not None and seg_out is not None:
        H, W = crop.shape[:2]
        cl = (seg_src[..., CLOTHES] > 0.6) & ~cv2.dilate((old_hair(seg_src, lm) > 0.4).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        Minv = cv2.invertAffineTransform(M)
        out_s = cv2.warpAffine(out, Minv, (W, H), borderMode=cv2.BORDER_REPLICATE)
        hair_o = cv2.warpAffine(np.ascontiguousarray(seg_out[..., HAIR]), Minv, (W, H)) > 0.3
        cl &= ~cv2.dilate(hair_o.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        # (with KEEP_CLOTHES her real clothes go on the result anyway, so no retry is spent on them)
        if cl.sum() > 2000 and not KEEP_CLOTHES:
            la = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
            lb = cv2.cvtColor(out_s, cv2.COLOR_RGB2LAB).astype(np.float32)
            # colour of the garment (broad): allow the lighting to change a little, not the colour
            dab = float(np.linalg.norm(la[cl][:, 1:].mean(0) - lb[cl][:, 1:].mean(0)))
            dL = abs(float(la[cl][:, 0].mean() - lb[cl][:, 0].mean()))
            # pattern: a plain garment must not come back printed (camouflage, florals), nor the reverse
            # one-sided and with a wide margin, so light, shadows and stray strands never count
            dtex = float(lb[cl][:, 0].std() - la[cl][:, 0].std()) + 0.5 * float(
                lb[cl][:, 1:].std(0).mean() - la[cl][:, 1:].std(0).mean())
            if dtex > 14:
                score += (dtex - 14) * 0.5 + 6
                why.append("her clothes were changed or a pattern was added; keep exactly the same garment, "
                           "colour and pattern as in image 1")
            if dab > 9 or dL > 30:
                score += max(dab - 9, 0) + max(dL - 30, 0) * 0.5 + 5
                why.append("her clothes changed colour; keep her clothing exactly as in image 1")
        # the strict clothes guardrail: her garment must still be there, in the same shape and place
        fit = garment_fit(crop, seg_src, out, seg_out, M)
        if fit is not None and fit[2] < 0.5 and not KEEP_CLOTHES:
            score += 20 + (0.5 - fit[2]) * 40
            why.append("her clothes, pose or body were changed; this is an edit of image 1: keep her exact clothes, "
                       "pose, arms and hands and change only the hair and the background")
        gap = skin_tone_gap(crop, seg_src, out, seg_out)
        if gap is not None and SKIN_TONE <= 0:  # otherwise her tone is put back afterwards anyway
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


async def ai_edit(crop: np.ndarray, wig_rgb: np.ndarray, prompt: str, keep: np.ndarray | None = None,
                  quality: str | None = None) -> np.ndarray:
    kwargs = dict(
        model=MODEL,
        image=[("customer.png", encode(crop), "image/png"), ("wig.png", encode(wig_rgb), "image/png")],
        prompt=prompt,
        size=f"{OUT_W}x{OUT_H}",
        quality=quality or QUALITY,
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


async def blend_pass(comp: np.ndarray, sent: np.ndarray, lm: np.ndarray, seg_src, wig_rgb: np.ndarray,
                     p: dict, outfit: str) -> np.ndarray:
    """AI retouch of the assembled result; returns the assembled one if anything goes wrong."""
    try:
        c_lm = await landmarks(comp)
        c_seg = await segment(comp)
        if c_lm is None or c_seg is None:
            return comp
        H, W = comp.shape[:2]
        inner = max(6, int(0.02 * W))
        body = (np.clip(c_seg[..., CLOTHES] + c_seg[..., BODY_SKIN], 0, 1) > 0.5).astype(np.uint8)
        body = cv2.erode(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner + 1,) * 2))
        hair = cv2.dilate((c_seg[..., HAIR] > 0.3).astype(np.uint8), np.ones((inner, inner), np.uint8))
        keep = np.maximum((body * (1 - hair)) * 255, keep_mask(comp.shape, c_lm, shrink=0.85)).astype(np.uint8)
        out2 = await ai_edit(comp, wig_rgb, build_prompt_blend(p, outfit), keep=keep, quality=BLEND_QUALITY)
        o_lm = await landmarks(out2)
        o_seg = await segment(out2)
        M2 = align(lm, o_lm) if o_lm is not None else None
        if M2 is None or o_seg is None:
            log.info("blend pass: result not usable, keeping the assembled picture")
            return comp
        if SKIN_TONE > 0 and seg_src is not None:
            out2 = keep_skin_tone(sent, seg_src, out2, o_seg, SKIN_TONE, exact=KEEP_FACE)
        if KEEP_FACE:
            out2, _ = face_lock(sent, lm, out2, o_lm, bool(p.get("bangs")), enhance=0, relight=0.0)
        if KEEP_CLOTHES:
            out2, how = keep_body(sent, seg_src, out2, o_seg, M2, inset=max(4, int(0.012 * W)))
            log.info("blend pass: her clothes %s", how)
            # No hands she isn't showing: skin the AI added where her photo has none is taken out
            skin_w = cv2.warpAffine(np.ascontiguousarray(seg_src[..., BODY_SKIN]), M2, (W, H))
            inside = cv2.warpAffine(np.ones(seg_src.shape[:2], np.float32), M2, (W, H)) > 0.99
            out2, gone = no_new_hands(out2, o_seg, skin_w, inside)
            if gone:
                log.info("blend pass: removed %d px of hands/arms the AI added", gone)
        return out2
    except HTTPException:
        log.warning("blend pass failed at the AI; keeping the assembled picture")
        return comp
    except Exception:
        log.exception("blend pass failed; keeping the assembled picture")
        return comp


async def final_gate(out: np.ndarray, lm: np.ndarray, seg_src, p: dict) -> None:
    try:
        f_lm = await landmarks(out)
        f_seg = await segment(out)
    except Exception:
        return
    if f_lm is None or f_seg is None:
        return
    problems = []
    M = align(lm, f_lm, strict=False)
    if seg_src is not None and M is not None and added_hands(seg_src, f_seg, M).any():
        problems.append("hands")
    if head_gaps(f_lm, f_seg, shoulders=False):  # the head itself must be covered (shoulders may vary)
        problems.append("head not covered")
    if bare_scalp(f_lm, f_seg):
        problems.append("bare scalp")
    if problems:
        log.warning("quality gate stopped this result: %s", ", ".join(problems))
        raise HTTPException(422, "We couldn't get a perfect result from this photo. Please take a new photo facing the "
                                 "camera in good light, with your head and shoulders in view and nothing in front of "
                                 "your face, and try again.")


async def to_her_frame(out: np.ndarray, lm: np.ndarray, sent: np.ndarray, seg_src) -> np.ndarray:
    try:
        f_lm = await landmarks(out)
        M = align(lm, f_lm) if f_lm is not None else None
        if M is None and f_lm is not None:
            M = align(lm, f_lm, strict=False)
        if M is None:
            return out
        H, W = sent.shape[:2]
        scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        if abs(scale - 1) < 0.01 and float(np.abs(M[:, 2]).max()) < 2:
            return out  # already in her frame
        Minv = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(out, Minv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        have = cv2.warpAffine(np.ones(out.shape[:2], np.float32), Minv, (W, H)) > 0.999
        if not have.all():
            # edges the AI's picture doesn't reach: her own photo there (her body), backdrop where she had none
            fill = np.empty_like(sent)
            fill[:] = BACKDROP_RGB
            if seg_src is not None:
                mine = np.clip(seg_src[..., BODY_SKIN] + seg_src[..., CLOTHES] + seg_src[..., HAIR], 0, 1)[..., None]
                fill = (sent * mine + fill * (1 - mine)).astype(np.uint8)
            a = cv2.GaussianBlur((~have).astype(np.float32), (0, 0), 2.0)[..., None]
            back = np.clip(back * (1 - a) + fill * a, 0, 255).astype(np.uint8)
        log.info("put back into her frame (the AI had scaled her by %.2f)", scale)
        return back
    except Exception:
        log.exception("frame alignment skipped")
        return out


async def fill_head_gaps(out: np.ndarray, lm: np.ndarray, wig_rgb: np.ndarray, p: dict) -> np.ndarray:
    try:
        f_lm = await landmarks(out)
        f_seg = await segment(out)
        if f_lm is None or f_seg is None:
            return out
        miss, area = head_gaps(f_lm, f_seg, want_mask=True, shoulders=is_long(p))
        bald, top = bare_scalp(f_lm, f_seg, want_mask=True)
        if bald:
            miss = (miss + " and " if miss else "") + "top of the head (bare scalp)"
            area = np.maximum(area, top)
        if not miss or not area.any():
            return out
        log.info("wig did not cover the %s: adding the wig's hair there", miss)
        desc = p.get("description") or p["name"]
        prompt = ("Image 2 is the wig she is wearing (" + desc + "). In the editable area beside her head, add the same "
                  "wig hair so the wig covers her whole head like a real, properly installed wig: the same braids, "
                  "colours and style, parted in the middle, lying flat on the scalp from the centre part, falling over "
                  "BOTH shoulders, covering the ear and falling down "
                  "naturally, joined seamlessly to the hair that is already there. No bare scalp, gap or background "
                  "around her head. Do not change her face, skin, clothes or anything outside the editable area. "
                  "Keep the plain " + BACKDROP_WORDS + " backdrop. Photorealistic.")
        fixed = await ai_edit(out, wig_rgb, prompt, keep=((1 - area) * 255).astype(np.uint8))
        W = out.shape[1]
        a = cv2.GaussianBlur(area.astype(np.float32), (0, 0), max(2.0, 0.006 * W))[..., None]
        return np.clip(out * (1 - a) + fixed * a, 0, 255).astype(np.uint8)
    except Exception:
        log.exception("head gap fill skipped")
        return out


async def remove_added_hands(out: np.ndarray, lm: np.ndarray, seg_src, wig_rgb: np.ndarray) -> np.ndarray:
    """Find hands/arms in the result that are not in her photo and have the AI repaint just those areas
    (continuing her clothes and the backdrop). Everything outside them is left exactly as it is."""
    try:
        f_lm = await landmarks(out)
        f_seg = await segment(out)
        Mf = align(lm, f_lm, strict=False) if f_lm is not None else None
        if Mf is None or f_seg is None:
            return out
        hands = added_hands(seg_src, f_seg, Mf)
        if not hands.any():
            return out
        W = out.shape[1]
        area = cv2.dilate(hands, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.04 * W) | 1,) * 2))
        log.info("removing %d px of added hands/arms", int(hands.sum()))
        prompt = ("Remove the hands and arms in the editable area completely. In their place continue what is around "
                  "them: her own clothes (same garment, colour and pattern) and the plain " + BACKDROP_WORDS +
                  " backdrop. Do not add any hand, arm or object. Keep the wig hair as it is. Everything outside the "
                  "editable area stays exactly the same. Photorealistic, seamless.")
        fixed = await ai_edit(out, wig_rgb, prompt, keep=((1 - area) * 255).astype(np.uint8), quality=BLEND_QUALITY)
        a = cv2.GaussianBlur(area.astype(np.float32), (0, 0), max(2.0, 0.006 * W))[..., None]
        return np.clip(out * (1 - a) + fixed * a, 0, 255).astype(np.uint8)
    except Exception:
        log.exception("hand removal skipped")
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
        out = await ai_edit(sent, wig_rgb, build_prompt_pure(p, note, outfit),
                            keep=None if FACE_SOURCE == "ai" else face_keep)  # ai: the face may be retouched
        out_lm = await landmarks(out)
        seg_out = None
        if seg_src is not None:
            try:
                seg_out = await segment(out)
            except Exception:
                seg_out = None
        score, why = check_result(crop, lm, out, out_lm, seg_src, seg_out, long_hair=is_long(p))
        log.info("try-on attempt %d: score %.1f %s", tries, score, why or "(passed)")
        if best is None or score < best[0]:
            best = (score, out, seg_out, out_lm, why)
        if score < 1.0:
            break
        note = why
    score, out, seg_out, out_lm, why = best
    # Her face stays her face: her own (smoothed) face goes back on the result, and the result's skin
    # (neck, ears, chest) is first brought to her exact skin colour so there is no edge where they meet.
    locked = False
    if SKIN_TONE > 0 and seg_src is not None and seg_out is not None:
        gap = skin_tone_gap(sent, seg_src, out, seg_out)
        out = keep_skin_tone(sent, seg_src, out, seg_out, SKIN_TONE, exact=KEEP_FACE)
        log.info("skin tone kept (was off by L %.1f, colour %.1f)", *(gap or (0, 0)))
    # The AI's lightly retouched face is used only when it is provably still her face; otherwise hers goes back
    face_ok = FACE_SOURCE == "ai" and out_lm is not None and "facial features changed" not in why \
        and "moved or changed size" not in why and "made smaller or bigger" not in why
    likeness = face_match(sent, lm, out, out_lm) if face_ok else 0.0
    if face_ok and likeness < FACE_MATCH_MIN:
        face_ok = False  # features are in place but don't look like hers closely enough
    if KEEP_FACE and not face_ok:
        # her real face, with a light smoothing so the result is still retouched
        mine = sent if FACE_ENHANCE > 0 else beautify(sent, lm, FALLBACK_SMOOTH, style="natural", skin_extra=body)
        out, locked = face_lock(mine, lm, out, out_lm, bool(p.get("bangs")), enhance=0, relight=0.0)
        log.info("her own face kept (AI face likeness %.2f): %s", likeness, locked)
    else:
        log.info("AI's lightly retouched face used (likeness %.2f)", likeness)
    # Her clothes stay her clothes, by construction: her real body and clothes from her photo, placed
    # relative to her face exactly as in her photo, under the new hair; anything the AI invented
    # outside her body becomes backdrop.
    if KEEP_CLOTHES and seg_src is not None and seg_out is not None:
        M = align(lm, out_lm) if out_lm is not None else None
        if M is None and out_lm is not None:
            M = align(lm, out_lm, strict=False)
        if M is not None:
            # How far the AI moved away from her real body/clothes. Close (most photos): her clothes go
            # on with invisible seams and no extra AI call. Far (it re-posed her): one AI blend pass.
            fit = garment_fit(sent, seg_src, out, seg_out, M)
            needs_blend = fit is not None and fit[2] < BLEND_BELOW
            # Always her real pose: her arms and hands exactly where they are in her photo. Whatever the AI
            # drew elsewhere (moved hands, extra arms) becomes backdrop; the blend pass smooths the result.
            out, how = keep_body(sent, seg_src, out, seg_out, M,
                                 valid=None if added is None else ~added,
                                 no_paste=neck_band(out.shape, out_lm) if face_ok else None)
            how += f", match {fit[2]:.2f}" if fit is not None else ""
            # Also blend when a visible amount of what the AI drew had to be taken out (that leaves hard
            # cut-out edges which only the blend pass can make natural)
            # (only a real re-pose triggers the blend pass; small trimmed edges are fine without it)
        else:
            out, how = restore_clothes(crop, seg_src, out, seg_out, None)
            if CLOTHES_STRICT and not (how.startswith("kept") or how.startswith("no clothes")):
                raise HTTPException(422, "We couldn't keep your outfit exactly as it is in this photo. Please take a "
                                         "new photo facing the camera, with your head and shoulders in view.")
        log.info("her own clothes: %s", how)
        # One AI retouch over the assembled picture so hair, body and backdrop become one photo (her face
        # and the inside of her clothes are protected, and both are put back exactly afterwards).
        if BLEND_PASS and M is not None and needs_blend:
            out = await blend_pass(out, sent, lm, seg_src, wig_rgb, p, outfit)
    # Deliver exactly what she gave: the result is put back into HER photo's frame, so her body is the same
    # size and in the same place as in her photo, and the picture ends where her photo ends (a half photo
    # gives a half result, the wig running off the edge). Anything the AI drew beyond her frame is dropped.
    out = await to_her_frame(out, lm, sent, seg_src)
    # The wig must cover her whole head: if a side or the top is still bare, the AI adds the wig's hair there
    out = await fill_head_gaps(out, lm, wig_rgb, p)
    # Never hands she isn't showing: if any are still in the picture, the AI repaints only those areas
    if seg_src is not None and BLEND_PASS:  # (without the blend pass her own body already covers this)
        out = await remove_added_hands(out, lm, seg_src, wig_rgb)
    # Safety net: no floating pieces of clothing or skin left anywhere on the finished picture
    if seg_src is not None:
        try:
            seg_f = await segment(out)
            if seg_f is not None:
                out, pieces = tidy_body(out, seg_f)
                if pieces:
                    log.info("tidied %d stray piece(s)", pieces)
                # the backdrop in the exact brand colour (a soft colour shift, hair edges blend into it)
                out = match_backdrop(out, seg_f, BACKDROP_RGB)
        except Exception:
            log.exception("tidy step skipped")
    # Brand safety gate: a result with someone else's hands or a visibly unfinished wig is never shown
    if QUALITY_GATE:
        await final_gate(out, lm, seg_src, p)
    before = crop
    if added is not None and added.any():
        a_ = cv2.GaussianBlur(added.astype(np.float32), (0, 0), 3)[..., None]
        before = np.clip(crop * (1 - a_) + np.full((OUT_H, OUT_W, 3), BACKDROP_RGB, np.float32) * a_, 0, 255).astype(np.uint8)
    mode = f"pure ai ({tries} attempt{'s' if tries > 1 else ''}, check score {score:.1f})"
    log.info("try-on finished: mode=%s", mode)
    return {"before": before, "after": out, "locked": True, "mode": mode}


async def run_tryon(rgb: np.ndarray, p: dict) -> dict:
    # 1. Find the face and crop a head-and-shoulders portrait at the output aspect ratio.
    lm = await landmarks(rgb)
    if lm is None:
        raise HTTPException(422, "We couldn't find a face in your photo. Please retake it.")
    # Close selfies: extend the canvas so there's room for the full wig above the head and the shoulders
    rgb, lm, valid = make_room(rgb, lm, OUT_W / OUT_H, fill=BACKDROP_RGB)
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
        before = np.clip(crop * (1 - a_) + np.full((OUT_H, OUT_W, 3), BACKDROP_RGB, np.float32) * a_, 0, 255).astype(np.uint8)
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
