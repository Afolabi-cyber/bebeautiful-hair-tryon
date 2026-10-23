"""Face landmarks, edit masks and the face lock that pastes the user's real face back.

Everything works on RGB uint8 numpy arrays and (N, 2) float landmark arrays in pixels.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODELS_DIR = os.getenv("MODELS_DIR") or os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
SEG_PATH = os.path.join(MODELS_DIR, "selfie_multiclass_256x256.tflite")
# Classes of the multiclass selfie segmenter
BG, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)

# MediaPipe face mesh indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
UPPER_LIDS = [159, 386]
FOREHEAD, CHIN = 10, 152
# Points that rarely move when only the hair changes: eye corners, nose, mouth, chin, brows.
STABLE = [33, 133, 362, 263, 1, 4, 6, 168, 61, 291, 13, 14, 152, 70, 300, 105, 334, 199, 175]
# Feature outlines the retouch must never soften: eyes, brows, lips.
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSTRILS = [98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64]


def _lab(rgb: np.ndarray) -> np.ndarray:
    """RGB (0..255, any dtype) -> Lab in float32 on OpenCV's 8-bit scale (L 0..255, a/b centred on 128),
    computed in full precision so smooth skin gradients never get stepped into visible contour lines."""
    lab = cv2.cvtColor(np.clip(np.asarray(rgb, np.float32) / 255.0, 0, 1), cv2.COLOR_RGB2LAB)
    lab[..., 0] *= 255.0 / 100.0
    lab[..., 1:] += 128.0
    return lab


def _rgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of _lab: float32 RGB 0..255 (no rounding)."""
    x = np.asarray(lab, np.float32).copy()
    x[..., 0] *= 100.0 / 255.0
    x[..., 1:] -= 128.0
    return np.clip(cv2.cvtColor(x, cv2.COLOR_LAB2RGB), 0, 1) * 255.0


def _to_u8(img: np.ndarray, seed: int = 7) -> np.ndarray:
    """Round to 8-bit with a tiny dither, so gradients don't band."""
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.uniform(-0.5, 0.5, img.shape).astype(np.float32), 0, 255).astype(np.uint8)


def ensure_model(url: str = MODEL_URL, path: str = MODEL_PATH, min_bytes: int = 500_000) -> str:
    """Download a model once. A partial or blocked download (tiny file, HTML error page) is removed
    so it is retried next time instead of failing forever."""
    if os.path.exists(path) and os.path.getsize(path) < min_bytes:
        os.remove(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) < min_bytes:
            os.remove(tmp)
            raise RuntimeError(f"download of {os.path.basename(path)} was blocked or incomplete")
        os.replace(tmp, path)
    return path


def _model_bytes(url: str, path: str) -> bytes:
    # Loading from bytes avoids MediaPipe path problems on Windows (spaces, non-ASCII user names).
    with open(ensure_model(url, path), "rb") as f:
        return f.read()


class FaceTool:
    """Thin wrapper over MediaPipe Face Landmarker (image mode, one face)."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_model_bytes(MODEL_URL, MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray | None, float]:
        """(landmarks, eyes-closed score 0..1 from the model's blink detector)."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect(img)
        if not res.face_landmarks:
            return None, 0.0
        h, w = rgb.shape[:2]
        lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)
        blink = 0.0
        try:
            sc = {c.category_name: c.score for c in res.face_blendshapes[0]}
            blink = min(sc.get("eyeBlinkLeft", 0.0), sc.get("eyeBlinkRight", 0.0))
        except Exception:
            pass
        return lm, float(blink)

    def landmarks(self, rgb: np.ndarray) -> np.ndarray | None:
        return self.detect(rgb)[0]


class Segmenter:
    """MediaPipe multiclass selfie segmenter: background, hair, body skin, face skin, clothes, others."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_buffer=_model_bytes(SEG_URL, SEG_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._seg = vision.ImageSegmenter.create_from_options(opts)

    def probs(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 6) float32 class probabilities at the image's own size."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        masks = self._seg.segment(img).confidence_masks
        h, w = rgb.shape[:2]
        return np.stack([cv2.resize(m.numpy_view().astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                         for m in masks], -1)


def align(src_lm: np.ndarray, out_lm: np.ndarray | None, strict: bool = True):
    """Similarity transform from the original photo to the generated one, or None if unreliable.

    strict=False accepts a looser fit (used to position the new hair, where a few pixels don't matter).
    """
    if out_lm is None:
        return None
    M, _ = cv2.estimateAffinePartial2D(src_lm[STABLE], out_lm[STABLE], method=cv2.LMEDS)
    if M is None:
        return None
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    mapped = src_lm[STABLE] @ M[:, :2].T + M[:, 2]
    err = float(np.median(np.linalg.norm(mapped - out_lm[STABLE], axis=1)))
    limit = 0.04 if strict else 0.15
    if not (0.5 < scale < 2.0) or err > limit * geom(out_lm).height:
        return None
    return M


def _wig_pixels(wig_rgb: np.ndarray) -> np.ndarray:
    """uint8 mask of the wig in its product photo (plain background read from the border)."""
    lab = cv2.cvtColor(wig_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[:8].reshape(-1, 3), lab[-8:].reshape(-1, 3), lab[:, :8].reshape(-1, 3), lab[:, -8:].reshape(-1, 3)])
    m = (np.linalg.norm(lab - np.median(border, axis=0), axis=-1) > 22).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def wig_shape(wig_rgb: np.ndarray) -> dict | None:
    """Outline of the wig plus where a face sits in it: crown top, face-opening width and centre."""
    m = _wig_pixels(wig_rgb)
    ys, xs = np.nonzero(m)
    if len(ys) < 1000:
        return None
    top, h = int(ys.min()), int(ys.max() - ys.min())
    gaps = []
    for y in range(int(top + 0.08 * h), int(top + 0.25 * h)):
        row = m[y]
        on = np.nonzero(row)[0]
        if len(on) < 2:
            continue
        d = np.diff(np.concatenate([[1], row, [1]]).astype(int))
        runs = [(a, b) for a, b in zip(np.nonzero(d == -1)[0], np.nonzero(d == 1)[0]) if a > on.min() and b < on.max()]
        if runs:
            a, b = max(runs, key=lambda t: t[1] - t[0])
            gaps.append((b - a, (a + b) / 2))
    if not gaps:  # no visible face opening: fall back to the wig's centre and a third of its width
        return {"mask": m, "top": top, "gap_w": float(np.ptp(xs)) / 3, "gap_cx": float(xs.mean())}
    return {"mask": m, "top": top, "gap_w": float(np.median([g[0] for g in gaps])),
            "gap_cx": float(np.median([g[1] for g in gaps]))}


def hair_zone(src_shape, src_lm: np.ndarray, shape: dict | None, bangs: bool = False) -> np.ndarray:
    """The only area the AI may change: the wig's product outline placed on the customer's head,
    plus a margin and the whole top of the head, minus the face. uint8 0/1, image frame.

    Everything outside it (centre of the chest, background, the rest of the clothes, the face) is
    always copied from the customer's original photo.
    """
    H, W = src_shape[:2]
    g = geom(src_lm)
    zone = np.zeros((H, W), np.uint8)
    if shape is not None:
        s = g.width / shape["gap_w"]
        cx = float(src_lm[FACE_OVAL][:, 0].mean())
        crown_y = float(src_lm[FOREHEAD][1]) - 0.45 * g.height
        M = np.float32([[s, 0, cx - s * shape["gap_cx"]], [0, s, crown_y - s * shape["top"]]])
        zone = cv2.warpAffine(shape["mask"], M, (W, H), flags=cv2.INTER_NEAREST)
        # Close small holes inside each curtain of hair, then add a margin for imperfect placement.
        zone = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        grow = max(3, int(0.14 * g.width))
        zone = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    else:  # unknown outline: sides of the head down to the shoulders
        cv2.ellipse(zone, tuple(np.round(src_lm[FOREHEAD]).astype(int)), (int(1.3 * g.width), int(1.6 * g.height)), 0, 0, 360, 1, -1)
    # The whole top of the head and the hairline are always editable.
    zone[head_area(src_shape, src_lm) > 0.5] = 1
    zone[keep_mask(src_shape, src_lm, bangs) > 0] = 0
    zone[above_skull(src_shape, src_lm) > 0.5] = 0  # the wig hugs the head: no hair floating above it
    return zone


def head_area(src_shape, src_lm: np.ndarray) -> np.ndarray:
    """float 0/1 ellipse over the top of the head: crown, hairline, temples and ears."""
    H, W = src_shape[:2]
    g = geom(src_lm)
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, tuple(np.round(src_lm[FOREHEAD] + g.up * 0.1 * g.height).astype(int)),
                (int(1.05 * g.width), int(0.85 * g.height)), 0, 0, 360, 1.0, -1)
    return m


def above_skull(src_shape, src_lm: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """float 0/1: the space ABOVE the head, outside the skull plus a wig's thickness (margin, in face
    heights). Hair is never allowed there, so the wig hugs the head instead of floating over it."""
    H, W = src_shape[:2]
    g = geom(src_lm)
    top = src_lm[FOREHEAD]
    skull = np.zeros((H, W), np.uint8)
    c = top - g.up * 0.02 * g.height
    ang = float(np.degrees(np.arctan2(-g.up[0], g.up[1])))  # tilt of the face axis
    cv2.ellipse(skull, tuple(np.round(c).astype(int)),
                (int((0.56 + 2.5 * margin) * g.width), int((0.52 + margin) * g.height)), ang, 0, 360, 1, -1)
    ys, xs = np.mgrid[0:H, 0:W]
    rel = (xs - top[0]) * g.up[0] + (ys - top[1]) * g.up[1]   # height above the top of the forehead
    above = (rel > 0.12 * g.height).astype(np.float32)
    return cv2.GaussianBlur(above * (1 - skull.astype(np.float32)), (0, 0), max(1.5, 0.004 * W))


def old_hair(seg_src: np.ndarray, src_lm: np.ndarray) -> np.ndarray:
    """float 0..1: the customer's own hair, plus anything worn ON the head (head wrap, bonnet, cap,
    an old wig's cap), which the hair model calls 'other'. Lower accessories (glasses, earrings,
    necklaces) are not included."""
    H, W = seg_src.shape[:2]
    g = geom(src_lm)
    top = src_lm[FOREHEAD]
    ys, xs = np.mgrid[0:H, 0:W]
    rel = (xs - top[0]) * g.up[0] + (ys - top[1]) * g.up[1]
    crown = (rel > -0.12 * g.height).astype(np.float32) * head_area(seg_src.shape, src_lm)
    crown[keep_mask(seg_src.shape, src_lm) > 0] = 0
    crown = cv2.GaussianBlur(crown, (0, 0), max(1.5, 0.006 * W))
    return np.maximum(seg_src[..., HAIR], seg_src[..., OTHERS] * crown)


def skin_lock(seg_src: np.ndarray, src_lm: np.ndarray, bangs: bool = False) -> np.ndarray:
    """uint8 0/1: the customer's real forehead and temple skin (and the face). The AI may not redraw it,
    so it must build the wig around her actual head: the wig's hairline meets her own hairline and the
    head keeps its real size and shape (no 'pasted face' seam). For fringe styles only the face below
    the eyes is locked, so the fringe can fall over the forehead."""
    H, W = seg_src.shape[:2]
    m = keep_mask(seg_src.shape, src_lm, bangs) > 0
    if not bangs:
        skin = (seg_src[..., FACE_SKIN] > 0.5).astype(np.uint8)
        skin = cv2.erode(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, int(0.012 * W)) + 1,) * 2))
        m |= (skin > 0) & (head_area(seg_src.shape, src_lm) > 0.5)
    return m.astype(np.uint8)


def photo_problem(shape, lm: np.ndarray, blink: float = 0.0) -> str | None:
    """A friendly reason to retake the photo, or None if it's usable."""
    H, W = shape[:2]
    g = geom(lm)
    eye_l, eye_r = lm[LEFT_EYE].mean(0), lm[RIGHT_EYE].mean(0)
    v = eye_r - eye_l
    roll = abs(float(np.degrees(np.arctan2(v[1], v[0]))))
    roll = min(roll, 180 - roll)
    if roll > 22:
        return "Your head is tilted. Hold the phone upright, sit or stand straight and look at the camera."
    mid = (lm[234] + lm[454]) / 2
    if abs(float(np.dot(lm[1] - mid, v / max(np.linalg.norm(v), 1e-6)))) > 0.22 * g.width:
        return "Please face the camera straight on."
    def ear(top, bot, a, b):
        return float(np.linalg.norm(lm[top] - lm[bot]) / max(np.linalg.norm(lm[a] - lm[b]), 1e-6))
    if blink > 0.55 or (ear(159, 145, 33, 133) + ear(386, 374, 362, 263)) / 2 < 0.13:
        return "Your eyes look closed. Please keep your eyes open and look at the camera."
    # The wig needs room: the whole head with space above it, and the neck and shoulders below.
    # Tight photos are extended automatically (make_room); only extreme close-ups are refused.
    if g.height > 0.55 * H:
        return "You're too close. Hold the phone at arm's length so your whole head, neck and shoulders are in the photo."
    if g.height < 0.1 * H:
        return "You're a bit far away. Move the phone a little closer."
    if float(lm[FOREHEAD][1]) < 0.02 * H or float(g.chin[1]) > 0.98 * H:
        return "Part of your face is cut off. Hold the phone a little further away so your whole head is in the photo."
    return None


def ear_area(src_shape, src_lm: np.ndarray) -> np.ndarray:
    """float 0/1: where the two ears sit, just outside the sides of the face."""
    H, W = src_shape[:2]
    g = geom(src_lm)
    side = src_lm[454] - src_lm[234]
    side = side / max(float(np.linalg.norm(side)), 1e-6)
    ang = float(np.degrees(np.arctan2(side[1], side[0])))
    m = np.zeros((H, W), np.float32)
    for idx, sgn in ((234, -1), (454, 1)):
        c = src_lm[idx] + sgn * side * 0.06 * g.width - g.up * 0.03 * g.height
        cv2.ellipse(m, tuple(np.round(c).astype(int)), (int(0.11 * g.width), int(0.19 * g.height)), ang, 0, 360, 1.0, -1)
    return m


def transfer_hair(src: np.ndarray, src_lm: np.ndarray, out: np.ndarray, out_lm: np.ndarray | None,
                  zone: np.ndarray, seg_src: np.ndarray | None = None, seg_out: np.ndarray | None = None,
                  enhance: float = 0.75, background: str = "studio", src_scale: float = 1.0) -> np.ndarray:
    """Final image = the customer's ORIGINAL photo, with the AI image used only inside `zone`
    (see hair_zone). If segmentation is available, only the actual new hair inside the zone is used,
    so clothes under and between the braids are the customer's too. Returns an image in src's frame.
    """
    H, W = src.shape[:2]
    M = align(src_lm, out_lm, strict=False)
    if M is None:  # assume the AI kept the framing
        M = np.float32([[out.shape[1] / W, 0, 0], [0, out.shape[0] / H, 0]])
    Minv = cv2.invertAffineTransform(M)
    to_src = lambda x: cv2.warpAffine(x, Minv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out_s = to_src(out).astype(np.float32)
    # The customer's photo is often enlarged to fill the frame (soft), while the AI draws crisp detail.
    # Soften the AI picture to the photo's real sharpness so hair and face look like ONE photo.
    if src_scale < 0.95:
        sig = float(np.clip(0.5 * np.sqrt(max(1 / src_scale ** 2 - 1, 0)), 0, 1.8))
        if sig > 0.25:
            out_s = cv2.GaussianBlur(out_s, (0, 0), sig)
    body_skin = None
    if seg_src is not None:
        body_skin = cv2.GaussianBlur(np.clip((seg_src[..., BODY_SKIN] - 0.3) / 0.4, 0, 1), (0, 0), max(2.0, 0.01 * W))
    base = beautify(src, src_lm, enhance, skin_extra=body_skin) if enhance > 0 else src.copy()
    z = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), max(1.5, 0.004 * W))  # soft fence edge

    if seg_out is None:
        final = base.astype(np.float32) * (1 - z[..., None]) + out_s * z[..., None]
        return np.clip(final, 0, 255).astype(np.uint8)

    p = np.clip(to_src(np.ascontiguousarray(seg_out[..., HAIR])), 0, 1)

    # Alpha for the new hair: firm inside, soft but not too wide at the edges (avoids dark halos
    # from the AI's background showing round the curls).
    a = np.clip((p - 0.35) / 0.4, 0, 1)
    a = a * a * (3 - 2 * a)

    # Gaps between braids/curls: pixels inside the hair area that look like the AI's surroundings
    # (its repainted clothes or background) rather than like the hair get the real photo instead.
    sig = max(2.0, 0.03 * W)
    def wmean(img, w):
        return cv2.GaussianBlur(img * w[..., None], (0, 0), sig) / np.maximum(cv2.GaussianBlur(w, (0, 0), sig), 1e-4)[..., None]
    lab = cv2.cvtColor(np.clip(out_s, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    around = wmean(lab, (a < 0.1).astype(np.float32))
    hairc = wmean(lab, (a > 0.9).astype(np.float32))
    d_bg = np.linalg.norm(lab - around, axis=-1)
    d_hair = np.linalg.norm(lab - hairc, axis=-1)
    on_clothes = (seg_src[..., CLOTHES] + seg_src[..., OTHERS] > 0.5) if seg_src is not None else np.zeros((H, W), bool)
    # only over the customer's real clothes, and only when the pixel is clearly fabric and clearly not hair
    gaps = (a > 0) & on_clothes & (d_bg < 15) & (d_hair > 30) & (d_bg < 0.4 * d_hair)
    gaps = cv2.morphologyEx(gaps.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    a[gaps] = 0
    # With the hair model the fence can be generous (the model already tells hair from clothes);
    # it only stops stray AI hair far away from the head.
    wide = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.04 * W) + 1,) * 2))
    a = cv2.GaussianBlur(a, (0, 0), 0.7) * cv2.GaussianBlur(wide.astype(np.float32), (0, 0), 3)

    # Their old hair that the wig doesn't cover (e.g. an afro sticking out at the sides) is filled
    # in from the surrounding background so it doesn't show next to the wig.
    # (all old hair is cleared, also under the wig, so the fill borrows from background, not hair)
    # The customer's ears: skin-coloured pixels in the ear areas. The hair model sometimes calls an ear
    # "hair" (next to short dark hair), so without this the ear would be cleared away with the old hair.
    face_in = keep_mask(src.shape, src_lm) > 0
    lab_src = cv2.cvtColor(base, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_ref = lab_src[face_in].mean(axis=0) if face_in.any() else lab_src.reshape(-1, 3).mean(axis=0)
    skin_like = np.clip((40 - np.linalg.norm(lab_src - skin_ref, axis=-1)) / 15, 0, 1)
    bg_src = seg_src[..., BG] if seg_src is not None else np.zeros((H, W), np.float32)
    ears = ear_area(src.shape, src_lm) * skin_like * np.clip((0.7 - bg_src) / 0.4, 0, 1)
    ears = cv2.GaussianBlur(cv2.morphologyEx(ears, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), (0, 0), 1.0)

    oh = old_hair(seg_src, src_lm) if seg_src is not None else np.zeros((H, W), np.float32)
    old = (oh > 0.4).astype(np.uint8) * 255
    old[_poly_mask(src.shape, src_lm[FACE_OVAL]) > 0.5] = 0  # never touch the face
    old[ears > 0.3] = 0                                      # or the ears
    if old.any():
        old = cv2.dilate(old, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, int(0.008 * W)) + 1,) * 2))
        base = cv2.inpaint(base, old, max(3, int(0.012 * W)), cv2.INPAINT_TELEA)

    # Flush fit: on the head itself (crown, hairline, temples, round the ears) the AI's picture is
    # used wherever the customer's old hair was or the AI drew hair, so the wig sits on the scalp the
    # way the AI seated it, with no strip of old hair, smudge or backdrop between wig and head.
    # Where the AI shows background there, the studio backdrop goes in (it is not "person").
    # The face below the brows stays the customer's own (keep_mask).
    head = head_area(src.shape, src_lm)
    head[keep_mask(src.shape, src_lm) > 0] = 0
    head = cv2.GaussianBlur(head, (0, 0), max(1.5, 0.006 * W))
    hair_src = np.clip((oh - 0.25) / 0.4, 0, 1)
    hair_src = cv2.dilate(hair_src, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, int(0.01 * W)) + 1,) * 2))
    not_bg_out = 1 - np.clip(to_src(np.ascontiguousarray(seg_out[..., BG])), 0, 1)
    # Old hair anywhere else (long hair or a wig she already wears, lying over the shoulders and chest)
    # is replaced by what the AI put there: the new wig, or her top/neck continued where the old hair
    # used to hide them (those pixels never existed in her photo). Without this they'd become holes.
    flush = np.maximum(head * np.maximum(hair_src, p), hair_src) * np.clip((not_bg_out - 0.3) / 0.4, 0, 1)
    flush = flush * (1 - ears)  # on the ears only the AI's actual hair is used (a wig falling over them)
    # Never AI skin over the customer's real skin (forehead, temples): that is what made the face look
    # cut out and pasted. AI hair over her skin (a hairline, braids at the temples) is still allowed.
    if seg_src is not None:
        real_skin = np.clip((seg_src[..., FACE_SKIN] + seg_src[..., BODY_SKIN] - 0.3) / 0.4, 0, 1)
        flush = flush * (1 - real_skin)
    a = np.maximum(a, cv2.GaussianBlur(flush, (0, 0), 1.0))

    # Any AI skin used at the hairline/temples is matched to the customer's own skin tone, so the
    # top of the forehead doesn't come out greyer or lighter than the rest of the face.
    skin_out = np.clip(to_src(np.ascontiguousarray(seg_out[..., FACE_SKIN] + seg_out[..., BODY_SKIN])), 0, 1)
    both = face_in & (skin_out > 0.5)
    if both.sum() > 200:
        lab_out = _lab(out_s)
        shift = lab_src[both].mean(axis=0) - lab_out[both].mean(axis=0)
        fixed = _rgb(lab_out + shift)
        out_s = out_s * (1 - skin_out[..., None]) + fixed * skin_out[..., None]

    a = a * (1 - above_skull(src.shape, src_lm))  # never hair floating above the head

    # Natural hair edges. Semi-transparent edge pixels of the AI hair still carry the AI's own
    # forehead/background colour, which shows as a light halo round the hairline and curls; pull
    # them toward the local hair colour. Then feather the hairline where it meets her skin a little,
    # like real baby hairs, instead of a hard cut-out edge.
    hp = (p > 0.85).astype(np.float32)
    hsig = max(2.0, 0.012 * W)
    hair_col = cv2.GaussianBlur(out_s * hp[..., None], (0, 0), hsig) / np.maximum(cv2.GaussianBlur(hp, (0, 0), hsig), 1e-4)[..., None]
    has_col = np.clip(cv2.GaussianBlur(hp, (0, 0), hsig) * 8, 0, 1)
    edge = np.clip((0.7 - p) / 0.65, 0, 1) * (p > 0.05) * has_col  # only the faint outer edge, not the strands
    out_s = out_s + (hair_col - out_s) * (edge * 0.6)[..., None]
    if seg_src is not None:
        skin_near = cv2.GaussianBlur(np.clip((seg_src[..., FACE_SKIN] - 0.3) / 0.4, 0, 1), (0, 0), max(2.0, 0.01 * W))
        a = a * (1 - skin_near) + cv2.GaussianBlur(a, (0, 0), max(1.0, 0.0025 * W)) * skin_near

    final = base.astype(np.float32) * (1 - a[..., None]) + out_s * a[..., None]

    if background in ("studio", "white") and seg_src is not None:
        # Keep the person (their clothes, skin, face and accessories) plus the new hair; replace the rest.
        # Everything that isn't background or their old hair counts as the person, so unusual garments
        # and accessories are always kept rather than cut away.
        person = np.clip(1 - seg_src[..., BG] - oh, 0, 1)
        person = np.maximum(person, _poly_mask(src.shape, src_lm[FACE_OVAL]))  # the face is always kept
        person = np.clip((person - 0.3) / 0.4, 0, 1)                          # crisper edge, no room halo
        person = np.maximum(person, ears)                                      # and so are the ears
        alpha = np.maximum(person, a)
        # Small slivers of backdrop trapped between the jaw/neck and the hair look like cut-out errors:
        # close them and fill them from the surrounding hair and skin.
        solid = (alpha > 0.5).astype(np.uint8)
        k = 2 * max(3, int(0.015 * W)) + 1
        holes = (cv2.morphologyEx(solid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0) & (solid == 0)
        # only right next to the face, jaw and neck (natural gaps between braids stay open)
        jaw = _poly_mask(src.shape, src_lm[FACE_OVAL])
        g0 = geom(src_lm)
        jaw = cv2.dilate((jaw > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.18 * g0.width) + 1,) * 2))
        holes &= jaw > 0
        if holes.any():
            hm = holes.astype(np.uint8) * 255
            final = cv2.inpaint(np.clip(final, 0, 255).astype(np.uint8), hm, max(3, int(0.01 * W)), cv2.INPAINT_TELEA).astype(np.float32)
            alpha = np.maximum(alpha, cv2.GaussianBlur(holes.astype(np.float32), (0, 0), 1.5))
        # A real studio photo has a soft edge, not a razor-sharp cut-out
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, 0.0018 * W))
        bd = backdrop(H, W, background)
        # (no "light wrap": against a white backdrop it reads as a pale outline round the face)
        final = final * alpha[..., None] + bd * (1 - alpha[..., None])
        made = np.maximum(a, 1 - alpha)  # pixels that didn't come from the camera (AI hair, backdrop)
    else:
        made = a
    # Matching camera grain: the AI hair and the backdrop are perfectly clean while the photo has
    # sensor noise; giving everything the same fine grain makes it read as one photograph.
    n = photo_noise(src, src_lm)
    if n > 0.3:
        rng = np.random.default_rng(int(src[::97, ::89].sum()) % (2 ** 32))
        g = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.55)
        g *= n / max(float(g.std()), 1e-6)
        retouched = (keep_mask(src.shape, src_lm) > 0).astype(np.float32) * min(1.0, enhance) * 0.6
        final += (g * np.maximum(made, retouched))[..., None]
    return np.clip(final, 0, 255).astype(np.uint8)


def clothes_keep(seg_src: np.ndarray, src_lm: np.ndarray) -> np.ndarray:
    """float 0..1: the customer's own clothes (and accessories worn below the face, like a necklace or
    a bag strap), not covered by her old hair. These pixels always come from her photo."""
    H, W = seg_src.shape[:2]
    g = geom(src_lm)
    ys, xs = np.mgrid[0:H, 0:W]
    below = ((xs - g.chin[0]) * g.up[0] + (ys - g.chin[1]) * g.up[1]) < 0.15 * g.height  # glasses etc. excluded
    c = seg_src[..., CLOTHES] + seg_src[..., OTHERS] * below
    c = np.clip((c - 0.35) / 0.35, 0, 1) * (1 - np.clip((old_hair(seg_src, src_lm) - 0.3) / 0.4, 0, 1))
    c[_poly_mask(seg_src.shape, src_lm[FACE_OVAL]) > 0.5] = 0
    return c


def clean_photo(src: np.ndarray, lm: np.ndarray, src_scale: float = 1.0) -> np.ndarray:
    """Remove camera noise and compression blotches from the face area of a noisy or enlarged selfie,
    with an edge-preserving denoiser (eyes, brows, lips and face shape stay sharp). Clean photos are
    left almost untouched. Without this, a dark/far-away selfie looks mottled and blotchy after retouch."""
    n = photo_noise(src, lm)
    up = max(0.0, 1.0 / max(src_scale, 1e-3) - 1.0)   # how much the photo was enlarged
    h = float(np.clip(1.0 * n + 2.0 * up, 0, 7))
    if h < 2.0:
        return src
    H, W = src.shape[:2]
    g = geom(lm)
    x0, x1 = int(max(0, lm[:, 0].min() - 0.6 * g.width)), int(min(W, lm[:, 0].max() + 0.6 * g.width))
    y0, y1 = int(max(0, lm[:, 1].min() - 0.8 * g.height)), int(min(H, lm[:, 1].max() + 0.8 * g.height))
    roi = np.ascontiguousarray(src[y0:y1, x0:x1])
    den = cv2.fastNlMeansDenoisingColored(roi, None, h, h * 0.8, 7, 21).astype(np.float32)
    w = np.ones(roi.shape[:2], np.float32)
    e = max(4, int(0.05 * min(roi.shape[:2])))
    w[:e] *= np.linspace(0, 1, e)[:, None]; w[-e:] *= np.linspace(1, 0, e)[:, None]
    w[:, :e] *= np.linspace(0, 1, e)[None]; w[:, -e:] *= np.linspace(1, 0, e)[None]
    out = src.copy()
    out[y0:y1, x0:x1] = np.clip(roi * (1 - w[..., None]) + den * w[..., None], 0, 255).astype(np.uint8)
    return out


def _clone_face(src: np.ndarray, src_lm: np.ndarray, ai: np.ndarray, seg_src: np.ndarray, seg_ai: np.ndarray,
                enhance: float, src_scale: float = 1.0) -> np.ndarray:
    """Her own face (lightly retouched) placed into the AI picture with NO visible edge.

    Her face keeps all of its own detail; only its broad light and colour are matched to the AI
    picture, and that match is measured on skin that is skin in BOTH pictures (never on hair or
    background, which is what bled white or dark lines into the cheeks before). The hand-over to the
    AI picture is wide and gradual and runs through plain skin, so there is no line to see.
    """
    H, W = src.shape[:2]
    g = geom(src_lm)
    fw = g.width
    body = None
    if seg_src is not None:
        body = cv2.GaussianBlur(np.clip((seg_src[..., BODY_SKIN] - 0.3) / 0.4, 0, 1), (0, 0), max(2.0, 0.01 * W))
    clean = clean_photo(src, src_lm, src_scale)
    face = (beautify(clean, src_lm, enhance, skin_extra=body) if enhance > 0 else clean).astype(np.float32)
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(r)) + 1,) * 2)

    hair_ai = cv2.dilate((seg_ai[..., HAIR] > 0.4).astype(np.uint8), ell(0.025 * fw)) > 0
    skin_a = (seg_ai[..., FACE_SKIN] + seg_ai[..., BODY_SKIN]) > 0.5
    skin_s = (seg_src[..., FACE_SKIN] + seg_src[..., BODY_SKIN]) > 0.5
    oval = _poly_mask(src.shape, src_lm[FACE_OVAL]) > 0.5
    core = keep_mask(src.shape, src_lm, shrink=0.88) > 0          # eyes, glasses, nose, mouth: always hers
    # Her face: the oval plus her real forehead (where she has skin in her photo AND the AI drew skin)
    region = (oval | (skin_s & (head_area(src.shape, src_lm) > 0.5))) & skin_a & ~hair_ai
    region |= core & ~hair_ai
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, ell(0.01 * fw))
    region = cv2.erode(region, ell(0.012 * fw))
    if region.sum() < 0.2 * g.width * g.height:
        return ai  # face mostly hidden: leave the AI picture (the identity lock already ran)

    # Broad light/colour difference, measured only where both pictures show plain skin
    both = (skin_a & skin_s & ~hair_ai).astype(np.float32)
    both[core] = 0  # not inside eyes/glasses/mouth, where AI and photo legitimately differ
    sig = max(4.0, 0.07 * fw)
    diff = ai - face
    wb = cv2.GaussianBlur(both, (0, 0), sig)
    local = cv2.GaussianBlur(diff * both[..., None], (0, 0), sig) / np.maximum(wb, 1e-4)[..., None]
    glob = diff[both > 0].mean(0) if both.sum() > 100 else np.zeros(3, np.float32)
    conf = np.clip(wb * 4, 0, 1)[..., None]
    offset = local * conf + glob * (1 - conf)
    offset = cv2.GaussianBlur(offset, (0, 0), sig * 0.5)
    matched = face + offset

    # Wide, gradual hand-over through skin
    soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(2.0, 0.02 * fw))
    soft = np.clip((soft - 0.05) / 0.9, 0, 1)
    soft = soft * soft * (3 - 2 * soft)
    soft = np.maximum(soft, cv2.GaussianBlur((core & ~hair_ai).astype(np.float32), (0, 0), 0.01 * fw))
    # Inside her real face outline, wherever the AI shows background instead of skin (its face was a
    # little narrower), her own face is used fully: background must never tint the face edge.
    # (only where her photo shows skin there, and only actual AI hair, not the safety margin round it, wins)
    bg_in_face = oval & (seg_ai[..., BG] > 0.3) & ~(seg_ai[..., HAIR] > 0.4) & skin_s
    if bg_in_face.any():
        # widened a little (the AI's background edge is only known to a pixel or two) but kept inside
        # her real skin, then feathered inward so no 1-pixel dip can let the background through
        cover = cv2.dilate(bg_in_face.astype(np.uint8), ell(0.03 * fw)).astype(bool) & oval & (skin_s | ~(seg_ai[..., BG] < 0.1))
        cover &= ~(seg_ai[..., HAIR] > 0.4)
        c = cv2.GaussianBlur(cover.astype(np.float32), (0, 0), 1.5)
        c = np.clip(c * 2, 0, 1) * cover + c * ~cover  # full strength inside, soft only outwards
        soft = np.maximum(soft, c)
    return ai * (1 - soft[..., None]) + matched * soft[..., None]


def _warp_field(H: int, W: int, src_pts: np.ndarray, dst_pts: np.ndarray, weight: np.ndarray, step: int = 8):
    """Smooth displacement field (inverse-distance weighted) so that a pixel at src_pts samples the
    picture at dst_pts. Returns float32 maps for cv2.remap, faded out where weight is 0."""
    gy, gx = np.mgrid[0:H:step, 0:W:step].astype(np.float32)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    d = dst_pts - src_pts
    acc = np.zeros_like(pts); wsum = np.zeros(len(pts), np.float32)
    for i in range(0, len(src_pts), 64):  # chunked to keep memory small
        sp, dd = src_pts[i:i + 64], d[i:i + 64]
        dist2 = ((pts[:, None, :] - sp[None]) ** 2).sum(-1) + 25.0
        w = 1.0 / dist2 ** 1.5
        acc += (w[..., None] * dd[None]).sum(1); wsum += w.sum(1)
    disp = (acc / wsum[:, None]).reshape(gx.shape + (2,))
    disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR)
    disp *= weight[..., None]
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return xs + disp[..., 0], ys + disp[..., 1]


def compose_ai(src: np.ndarray, src_lm: np.ndarray, out: np.ndarray, out_lm: np.ndarray | None,
               seg_src: np.ndarray, seg_out: np.ndarray, background: str = "studio",
               src_scale: float = 1.0, detail: float = 0.6, enhance: float = 0.75,
               added: np.ndarray | None = None) -> np.ndarray | None:
    """Natural blend: the AI renders the whole head (face, hairline, hair, light and shadows together,
    lightly beautified), then the face is put back into the customer's EXACT geometry and her real
    feature detail is restored, and her clothes always come from her own photo.

    Returns an image in src's frame, or None if the AI image can't be matched to the photo reliably
    (the caller then uses the photo-based composite instead).
    """
    if out_lm is None:
        return None
    H, W = src.shape[:2]
    M = align(src_lm, out_lm, strict=False)
    if M is None:
        return None
    g = geom(src_lm)
    fw = g.width
    # Safety: if the AI changed the framing (zoomed/shrank the head) or the photo is very tight, the face
    # would have to be stretched a lot to match; the caller then uses the safer photo-based blend.
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    if not (0.88 < scale < 1.12) or g.height > 0.42 * H:
        return None
    Minv = cv2.invertAffineTransform(M)
    to_src = lambda x, interp=cv2.INTER_LINEAR: cv2.warpAffine(x, Minv, (W, H), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    ai = to_src(out).astype(np.float32)
    ai_lm = out_lm @ Minv[:, :2].T + Minv[:, 2]

    # 1. Her exact face shape: move every facial point of the AI face back to where it is on her photo.
    d = ai_lm - src_lm
    if float(np.median(np.linalg.norm(d, axis=1))) > 0.05 * g.height or float(np.percentile(np.linalg.norm(d, axis=1), 90)) > 0.1 * g.height:
        return None  # the AI changed the face shape too much to correct without visible stretching
    lim = 0.1 * fw
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    d = d * np.minimum(1, lim / np.maximum(nrm, 1e-6))
    idx = np.arange(0, len(src_lm), 3)
    idx = np.unique(np.concatenate([idx, FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS, LEFT_BROW, RIGHT_BROW, NOSTRILS]))
    region = _poly_mask(src.shape, src_lm[FACE_OVAL])
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.12 * fw) + 1,) * 2))
    region = cv2.GaussianBlur(region, (0, 0), 0.08 * fw)
    mx, my = _warp_field(H, W, src_lm[idx], src_lm[idx] + d[idx], region)
    ai = cv2.remap(ai, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    seg_ai = cv2.remap(to_src(np.ascontiguousarray(seg_out)), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Same sharpness as her photo (it is often enlarged)
    if src_scale < 0.95:
        sig = float(np.clip(0.5 * np.sqrt(max(1 / src_scale ** 2 - 1, 0)), 0, 1.8))
        if sig > 0.25:
            ai = cv2.GaussianBlur(ai, (0, 0), sig)

    # Her real skin tone: the AI may brighten slightly for a glow, but never lighten, darken or grey her
    # skin. Any larger drift in tone is pulled back to her own (on all skin: face, neck, ears).
    face_core = cv2.erode((_poly_mask(src.shape, src_lm[FACE_OVAL]) > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    face_core &= seg_src[..., FACE_SKIN] > 0.4
    skin_ai = np.clip(seg_ai[..., FACE_SKIN] + seg_ai[..., BODY_SKIN], 0, 1)
    if face_core.sum() > 300:
        lab_s = _lab(src)
        lab_a = _lab(ai)
        ms, ma = lab_s[face_core].mean(0), lab_a[face_core].mean(0)
        dL = ma[0] - ms[0]
        fixL = -(dL - 6) if dL > 6 else (-(dL + 2) if dL < -2 else 0.0)   # allow up to +6 brighter
        fix = np.array([fixL, ms[1] - ma[1], ms[2] - ma[2]], np.float32)
        fix[1:] = np.where(np.abs(fix[1:]) > 2.5, fix[1:] - np.sign(fix[1:]) * 2.5, 0)  # small warmth change ok
        if np.abs(fix).max() > 0.5:
            w_sk = cv2.GaussianBlur(np.maximum(skin_ai, (_poly_mask(src.shape, src_lm[FACE_OVAL]) > 0.5) * 1.0), (0, 0), 0.03 * fw)
            lab_a += fix * w_sk[..., None]
            ai = _rgb(lab_a)

    # 2. HER face, every time. The AI draws a slightly different face on every run, so the inner face
    #    (brows down to the chin, cheek to cheek) always comes from her own photo, lightly retouched.
    #    Poisson blending ("seamless clone") keeps all of her real facial detail but takes on the colour
    #    and light of the AI picture at the edges, so it sits in the same light as the new hair with no seam.
    #    Braids the AI drew falling over the cheeks stay in front.
    ai = _clone_face(src, src_lm, ai, seg_src, seg_ai, enhance, src_scale)

    # Her real eyes, brows, nose and lips: put back their fine detail (not their colour or light).
    feats = np.zeros((H, W), np.float32)
    for ids, grow in ((LEFT_EYE, 0.05), (RIGHT_EYE, 0.05), (LEFT_BROW, 0.03), (RIGHT_BROW, 0.03),
                      (LIPS, 0.03), (NOSTRILS, 0.02)):
        feats = np.maximum(feats, _poly_mask(src.shape, src_lm[ids], grow=max(1, int(grow * fw)), blur=0.02 * fw))
    srcf = clean_photo(src, src_lm, src_scale).astype(np.float32)  # (noise-free detail)
    ts = max(1.0, 0.006 * fw)
    hf = (srcf - cv2.GaussianBlur(srcf, (0, 0), ts)) - (ai - cv2.GaussianBlur(ai, (0, 0), ts))
    ai += hf * (feats * detail)[..., None]

    # 3. Her clothes, always from her photo. New hair falling in front of them stays.
    hair = np.clip((seg_ai[..., HAIR] - 0.35) / 0.4, 0, 1)
    hair = hair * hair * (3 - 2 * hair)
    keep = clothes_keep(seg_src, src_lm) * (1 - hair)
    keep = cv2.GaussianBlur(cv2.erode(keep, np.ones((3, 3), np.uint8)), (0, 0), 0.8)
    final = ai * (1 - keep[..., None]) + srcf * keep[..., None]
    # Clothes the AI drew where she has none (e.g. it widened the shoulders) are never shown
    ai_clothes_only = (seg_ai[..., CLOTHES] > 0.5) & (clothes_keep(seg_src, src_lm) < 0.2) & (hair < 0.3)
    ai_clothes_only &= ~(old_hair(seg_src, src_lm) > 0.4)  # (under her old long hair the AI must fill in)
    added = np.zeros((H, W), bool) if added is None else added.astype(bool)
    ai_clothes_only &= ~added  # (beyond the edge of her photo the AI continues her shoulders)

    # 4. Background: clean studio backdrop with a soft, natural edge and a little light wrap.
    if background in ("studio", "white"):
        # Only the new hair, skin and her own clothes are "person"; leftovers the AI drew (bits of a hood,
        # room objects) go to the backdrop instead of floating round her.
        person = np.clip(seg_ai[..., HAIR] + seg_ai[..., FACE_SKIN] + seg_ai[..., BODY_SKIN], 0, 1)
        person = np.maximum(person, np.clip((clothes_keep(seg_src, src_lm) - 0.2) / 0.5, 0, 1))
        # where her old long hair hid her top, the AI's continuation of the top is part of her
        under_old = cv2.dilate((old_hair(seg_src, src_lm) > 0.4).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        person = np.maximum(person, (seg_ai[..., CLOTHES] + seg_ai[..., OTHERS]) * (under_old | added))
        person = np.maximum(person, keep)
        person = np.maximum(person, (_poly_mask(src.shape, src_lm[FACE_OVAL]) > 0.5) * np.clip(seg_src[..., FACE_SKIN] + seg_src[..., BODY_SKIN] + (1 - seg_ai[..., BG]), 0, 1))
        person[ai_clothes_only] = 0
        person = np.clip((person - 0.25) / 0.5, 0, 1)
        # Thin slivers of backdrop between her face/neck and the hair show as white lines: close them
        # and fill them from the hair and skin around them.
        solid = (person > 0.5).astype(np.uint8)
        kk = 2 * max(3, int(0.035 * W)) + 1
        holes = (cv2.morphologyEx(solid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))) > 0) & (solid == 0)
        near = cv2.dilate((_poly_mask(src.shape, src_lm[FACE_OVAL]) > 0.5).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.3 * fw) + 1,) * 2)) > 0
        ys_, xs_ = np.mgrid[0:H, 0:W]
        neck = ((xs_ - g.chin[0]) ** 2 / (0.7 * fw) ** 2 + (ys_ - g.chin[1] - 0.35 * g.height) ** 2 / (0.6 * g.height) ** 2) < 1
        holes &= (near | neck)
        if holes.any():
            # fill only from the person around it (hair and skin), never from the white backdrop
            fill = _masked_blur(final, (person > 0.6).astype(np.float32), max(3.0, 0.012 * W))
            hm = cv2.GaussianBlur(holes.astype(np.float32), (0, 0), 1.5)
            final = final * (1 - hm[..., None]) + fill * hm[..., None]
            # the filled area overlaps its neighbours a little, so no half-transparent seam is left between
            cover = cv2.dilate(holes.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
            person = np.maximum(person, cv2.GaussianBlur(cover, (0, 0), 1.0))
            # pixels right round the filled gap that still carry the backdrop's white (edges of strands)
            pale = np.clip((final.mean(-1) - fill.mean(-1) - 15) / 25, 0, 1) * cv2.GaussianBlur(cover, (0, 0), 1.0)
            final = final * (1 - pale[..., None]) + fill * pale[..., None]
        person = cv2.GaussianBlur(person, (0, 0), max(1.0, 0.0018 * W))
        bd = backdrop(H, W, background)
        # (no "light wrap": against a white backdrop it reads as a pale outline round the face)
        final = final * person[..., None] + bd * (1 - person[..., None])
        made = (1 - keep) * 1.0
    else:
        final[ai_clothes_only] = srcf[ai_clothes_only]
        made = 1 - keep

    # 5. One consistent camera grain over everything the AI made
    n = min(photo_noise(src, src_lm) * 0.5, 1.6)  # low: the face itself was cleaned
    if n > 0.3:
        rng = np.random.default_rng(int(src[::97, ::89].sum()) % (2 ** 32))
        gr = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.55)
        final += (gr * (n / max(float(gr.std()), 1e-6)) * made)[..., None]
    return _to_u8(final)


def photo_noise(src: np.ndarray, lm: np.ndarray) -> float:
    """Camera noise level of the photo (std of fine grain on the cheeks), for matching grain."""
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hf = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    m = cv2.erode(keep_mask(src.shape, lm), np.ones((9, 9), np.uint8)) > 0
    if m.sum() < 200:
        return 0.0
    v = hf[m]
    return float(np.clip(1.4826 * np.median(np.abs(v - np.median(v))), 0, 4.0))


def backdrop(H: int, W: int, kind: str = "studio") -> np.ndarray:
    """Clean photo-studio background: soft light gradient with a gentle glow behind the head."""
    if kind == "white":
        return np.full((H, W, 3), 255, np.float32)
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top, bottom = np.array([251, 247, 249], np.float32), np.array([236, 227, 232], np.float32)
    bg = top * (1 - t) + bottom * t
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow = np.exp(-(((xx - W / 2) / (0.55 * W)) ** 2 + ((yy - 0.35 * H) / (0.45 * H)) ** 2))[..., None]
    return np.clip(bg + glow * 6, 0, 255)


def keep_clothes(src: np.ndarray, out: np.ndarray, M: np.ndarray | None,
                 seg_src: np.ndarray, seg_out: np.ndarray) -> tuple[np.ndarray, float]:
    """Put the customer's original clothes back wherever the new hair isn't covering them.

    seg_src is the segmentation of `src` (original photo), seg_out of `out` (generated photo).
    Returns (image, fraction of the image restored).
    """
    H, W = out.shape[:2]
    if M is None:
        M = np.float32([[W / src.shape[1], 0, 0], [0, H / src.shape[0], 0]])  # same framing assumed
    warp = lambda x, interp=cv2.INTER_LINEAR: cv2.warpAffine(x, M, (W, H), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    src_w = warp(src)
    clothes = warp(seg_src[..., CLOTHES] + seg_src[..., OTHERS])  # garments and accessories in the original
    hair = seg_out[..., HAIR]
    # Grow the new hair a little so wispy curl ends and their soft edges stay the AI's.
    k = max(3, int(round(0.006 * W)) | 1)
    hair = cv2.dilate(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    clothes = np.clip(clothes, 0, 1)
    keep = clothes * (1 - np.clip(hair, 0, 1)) * (1 - seg_out[..., FACE_SKIN])
    keep = (keep > 0.5).astype(np.float32)

    # Gaps between braids/curls lying on the clothes: the segmenter marks them as hair, so the AI's
    # repainted fabric would show through. Where those pixels match the colour the AI gave the
    # clothes (and differ from the real fabric), use the real fabric instead.
    lab_o = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_s = cv2.cvtColor(src_w, cv2.COLOR_RGB2LAB).astype(np.float32)
    open_cloth = keep > 0.5
    zone = (clothes > 0.5) & ~open_cloth
    if open_cloth.sum() > 500 and zone.any():
        fake = np.median(lab_o[open_cloth], axis=0)
        spread = float(np.median(np.linalg.norm(lab_o[open_cloth] - fake, axis=1)))
        if spread < 18:  # the AI's clothing is a fairly uniform colour, so the test is reliable
            d_fake = np.linalg.norm(lab_o - fake, axis=-1)
            near_fake = d_fake < max(14.0, 2.5 * spread)
            changed = np.linalg.norm(lab_o - lab_s, axis=-1) > 20
            # Never take a pixel that looks like the wig itself (dark wigs on dark repaints).
            hair_only = (hair > 0.5) & (clothes < 0.3)
            if hair_only.sum() > 200:
                d_hair = np.linalg.norm(lab_o - np.median(lab_o[hair_only], axis=0), axis=-1)
                near_fake &= d_fake < 0.5 * d_hair
            gaps = zone & near_fake & changed
            gaps = cv2.morphologyEx(gaps.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            gap_keep = cv2.GaussianBlur(gaps.astype(np.float32), (0, 0), 0.6)  # tight edge: don't bleed into braids
        else:
            gap_keep = None
    else:
        gap_keep = None

    keep = cv2.GaussianBlur(keep, (0, 0), max(1.0, 0.003 * W))
    if gap_keep is not None:
        keep = np.maximum(keep, gap_keep)
    out = (src_w.astype(np.float32) * keep[..., None] + out.astype(np.float32) * (1 - keep[..., None]))
    return np.clip(out, 0, 255).astype(np.uint8), float(keep.mean())


@dataclass
class FaceGeom:
    up: np.ndarray       # unit vector pointing from chin to forehead
    height: float        # forehead to chin distance in px
    width: float         # cheek to cheek distance in px
    chin: np.ndarray


def geom(lm: np.ndarray) -> FaceGeom:
    v = lm[FOREHEAD] - lm[CHIN]
    h = float(np.linalg.norm(v))
    return FaceGeom(up=v / max(h, 1e-6), height=h, width=float(np.linalg.norm(lm[234] - lm[454])), chin=lm[CHIN])


def make_room(rgb: np.ndarray, lm: np.ndarray, aspect: float = 2 / 3):
    """Close selfies leave no room above the head (or below the chin) for a wig. Extend the canvas so a
    full head-and-shoulders frame fits; the extension is a soft, blurred continuation that the AI repaints
    (hair, top of the head, shoulders, backdrop). Returns (image, landmarks, valid) where valid is 1 on
    real photo pixels and 0 on the added area."""
    H, W = rgb.shape[:2]
    g = geom(lm)
    # Enough for the head with room above it, the hair and the shoulders; not a whole invented body.
    box_h = 3.4 * g.height
    box_w = box_h * aspect
    cx = float(lm[FACE_OVAL][:, 0].mean())
    top = float(lm[FOREHEAD][1]) - 1.2 * g.height
    pt = int(np.ceil(max(0.0, -top)))
    # Never extend below or far to the sides: that is where the AI would have to invent her shoulders,
    # arms and the rest of her clothes. Only room above the head and a little at the sides for the
    # hair; the crop then frames what she really photographed.
    pb = 0
    need_w = min(box_w, 2.8 * g.width)
    pl = int(np.ceil(max(0.0, need_w / 2 - cx)))
    pr = int(np.ceil(max(0.0, cx + need_w / 2 - W)))
    valid = np.ones((H, W), np.uint8)
    if max(pt, pb, pl, pr) == 0:
        return rgb, lm, valid
    # The added room is plain light backdrop (stretched edge pixels would give the AI streaks to copy)
    big = cv2.copyMakeBorder(rgb, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(226, 226, 226))
    valid = cv2.copyMakeBorder(valid, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    m = cv2.GaussianBlur((1 - valid).astype(np.float32), (0, 0), 6)[..., None]
    plain = np.full_like(big, 226)
    big = np.clip(big * (1 - m) + plain * m, 0, 255).astype(np.uint8)
    return big, lm + np.float32([pl, pt]), valid


def head_crop(shape: tuple[int, int], lm: np.ndarray, aspect: float = 2 / 3) -> tuple[int, int, int, int]:
    """A portrait (w:h = aspect) box around the head with headroom for big styles. Returns x, y, w, h."""
    H, W = shape[:2]
    g = geom(lm)
    box_h = 4.6 * g.height
    box_h = min(box_h, H, W / aspect)
    box_w = box_h * aspect
    cx = float(lm[FACE_OVAL][:, 0].mean())
    top = float(lm[FOREHEAD][1]) - 1.2 * g.height
    x = int(round(np.clip(cx - box_w / 2, 0, W - box_w)))
    y = int(round(np.clip(top, 0, H - box_h)))
    return x, y, int(round(box_w)), int(round(box_h))


def keep_mask(shape: tuple[int, int], lm: np.ndarray, bangs: bool = False, shrink: float = 0.92) -> np.ndarray:
    """uint8 mask (255 = the user's face we must not change).

    The face oval is shrunk a little so hair can frame the cheeks and temples, and everything
    above the brows (or above the eyes for fringe styles) is released so the hairline can change.
    """
    H, W = shape[:2]
    g = geom(lm)
    oval = lm[FACE_OVAL]
    c = oval.mean(axis=0)
    oval = c + (oval - c) * shrink
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(oval).astype(np.int32)], 255)

    rel = lambda pts: (pts - g.chin) @ g.up  # height of each point above the chin, along the face axis
    if bangs:
        cut = float(rel(lm[UPPER_LIDS]).max()) + 0.03 * g.height
    else:
        cut = float(rel(lm[BROWS]).max()) + 0.03 * g.height
    ys, xs = np.mgrid[0:H, 0:W]
    above = ((xs - g.chin[0]) * g.up[0] + (ys - g.chin[1]) * g.up[1]) > cut
    m[above] = 0
    return m


def openai_mask_png(keep: np.ndarray) -> bytes:
    """OpenAI edit mask: transparent = may edit, opaque = keep."""
    rgba = np.zeros((*keep.shape, 4), np.uint8)
    rgba[..., 3] = np.where(keep > 0, 255, 0)
    ok, buf = cv2.imencode(".png", rgba)
    assert ok
    return buf.tobytes()


def _poly_mask(shape, pts: np.ndarray, grow: int = 0, blur: float = 0) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [cv2.convexHull(np.round(pts).astype(np.int32))], 255)
    if grow > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return m.astype(np.float32) / 255.0


def beautify(rgb: np.ndarray, lm: np.ndarray, strength: float = 0.75, style: str | None = None,
             skin_extra: np.ndarray | None = None) -> np.ndarray:
    """Professional-style retouch that never moves or reshapes anything.

    Face recognition relies on the geometry and appearance of the eyes, brows, nose, mouth and
    face shape. All of those pixels stay where they are; the retouch only works on skin tone,
    texture and light:
      - smooths skin while keeping fine pore texture (frequency separation), so it doesn't look plastic
      - evens out blotches and redness without changing the overall skin tone
      - removes oily shine, gently lifts the face and the under-eye area
      - sharpens eyes, brows and lips slightly and adds a touch of lip colour
    """
    k = float(np.clip(strength, 0, 1))
    if k <= 0:
        return rgb
    H, W = rgb.shape[:2]
    fw = float(np.linalg.norm(lm[234] - lm[454]))
    px = lambda f: max(1, int(round(f * fw)))

    face = _poly_mask(rgb.shape, lm[FACE_OVAL], blur=px(0.02))
    feats = np.zeros((H, W), np.float32)
    for idx, grow in ((LEFT_EYE, 0.035), (RIGHT_EYE, 0.035), (LEFT_BROW, 0.02), (RIGHT_BROW, 0.02),
                      (LIPS, 0.02), (NOSTRILS, 0.0)):
        feats = np.maximum(feats, _poly_mask(rgb.shape, lm[idx], grow=px(grow), blur=px(0.012)))
    skin = face * (1 - feats)

    img = rgb.astype(np.float32)

    # 1. Skin smoothing by frequency separation (what professional retouchers use): blemishes and
    #    unevenness live in the middle frequencies, so only those are softened; the broad shading and
    #    the fine pore texture stay as they are. Blurs only average skin with skin (no halos from eyes,
    #    brows, lips or hair). This can't flatten skin into patches or steps the way an edge-preserving
    #    (bilateral) filter does on deep skin tones and enlarged photos.
    w = np.clip(skin, 1e-3, 1)
    s_hi, s_lo = max(0.8, 0.004 * fw), max(3.0, 0.025 * fw)
    hi_blur = _masked_blur(img, w, s_hi)
    lo_blur = _masked_blur(img, w, s_lo)
    fine = img - hi_blur
    mid = hi_blur - lo_blur
    smooth = lo_blur + mid * (1 - 0.45 * k) + fine
    a = skin[..., None]
    img = img * (1 - a) + smooth * a

    lab = _lab(img)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

    # 2. Even tone: pull local colour blotches toward the surrounding skin colour (overall tone unchanged).
    wsk = np.maximum(skin, 1e-3)
    wblur = np.maximum(cv2.GaussianBlur(wsk, (0, 0), px(0.05)), 1e-4)
    for ch in (A, B):
        local = cv2.GaussianBlur(ch * wsk, (0, 0), px(0.05)) / wblur
        ch += (local - ch) * skin * 0.55 * k

    inside = skin > 0.5
    if inside.sum() > 200:
        # 3. Shine: compress the brightest skin highlights.
        thr = float(np.percentile(L[inside], 85))
        L -= np.maximum(0, L - thr) * skin * 0.6 * k
        # 4. Soft lift on the whole face, a little more under the eyes.
        under = np.zeros((H, W), np.float32)
        for lid, corner_a, corner_b in ((145, 33, 133), (374, 263, 362)):
            c = lm[lid] + np.array([0, 0.055 * fw])
            axes = (int(np.linalg.norm(lm[corner_a] - lm[corner_b]) * 0.62), px(0.05))
            cv2.ellipse(under, tuple(np.round(c).astype(int)), axes, 0, 0, 360, 1.0, -1)
        under = cv2.GaussianBlur(under, (0, 0), px(0.03)) * (1 - feats)
        # natural = smoothing only: no lift, so the eyes and face keep their own light
        if (style or "").lower() != "natural":
            L += face * 4.0 * k + under * 6.0 * k

    lab = np.stack([L, A, B], -1)
    out = _rgb(lab)

    # 5. Crisper eyes, brows and lips; a touch of lip colour.
    # (not in natural: sharpening makes eyes look glassy, worst on small faces that were enlarged)
    feat_k = 0.0 if (style or "").lower() == "natural" else k
    sharp = out + (out - cv2.GaussianBlur(out, (0, 0), max(1.0, 0.006 * fw))) * 0.6 * feat_k
    fm = (feats * face)[..., None]
    out = out * (1 - fm) + sharp * fm
    lips = _poly_mask(rgb.shape, lm[LIPS], blur=px(0.008))
    lab = _lab(out)
    lab[..., 1:] = 128 + (lab[..., 1:] - 128) * (1 + lips * 0.12 * feat_k)[..., None]
    out = _rgb(lab)
    if (style or os.getenv("BEAUTY_STYLE", "glam")).lower() == "glam":
        out = glam(out, lm, k, face, skin, feats, px, skin_extra)
    # Put the real skin texture (pores, fine detail) back on top of the evened-out tone. Without it the
    # skin turns smooth and painted, which is the clearest sign of a filtered/AI image.
    o = out.astype(np.float32)
    src_f = rgb.astype(np.float32)
    ts = max(0.8, 0.0035 * fw)
    tex = (src_f - cv2.GaussianBlur(src_f, (0, 0), ts)) - (o - cv2.GaussianBlur(o, (0, 0), ts))
    o += tex * (skin * 0.7)[..., None]
    return _to_u8(o)


MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
CHEEKS = [(50, 205), (280, 425)]  # (cheekbone, lower cheek) on each side


def glam(rgb: np.ndarray, lm: np.ndarray, k: float, face, skin, feats, px, skin_extra=None) -> np.ndarray:
    """Beauty-filter look (like a Snapchat beauty lens) on top of the retouch, light and colour only:
    nothing is moved, enlarged or slimmed, so face recognition still matches the person.
      - extra-smooth, even, luminous skin with a soft glow
      - brighter, clearer eyes (whites and irises), whiter teeth
      - rosy glossy lips, a soft blush on the cheekbones, a warm healthy tone
    """
    H, W = rgb.shape[:2]
    img = rgb.astype(np.float32)

    # 1. A little more evening of mid-frequency unevenness (same frequency separation; texture kept).
    w = np.clip(skin, 1e-3, 1)
    hi_blur = _masked_blur(img, w, max(0.8, 0.004 * px(1)))
    lo_blur = _masked_blur(img, w, max(3.0, px(0.03)))
    img = img + (lo_blur - hi_blur) * (0.12 * k * skin)[..., None]

    # 2. Luminous skin: an exposure lift (multiplying R, G and B together keeps the skin's own colour
    #    and richness; adding brightness instead turns deep skin tones ashy and grey).
    # the neck and other visible skin get the same lift, so the face doesn't look brighter than the neck
    tone = face if skin_extra is None else np.maximum(face, np.clip(skin_extra, 0, 1))
    gain = 1 + tone * 0.11 * k
    img = 255 * (1 - (1 - np.clip(img / 255, 0, 1)) ** gain[..., None])  # lifts darker tones most, no clipping

    lab = _lab(img)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    # richer, warmer skin colour (a healthy glow, never grey)
    A += (A - 128) * tone * 0.1 * k + tone * 0.8 * k
    B += (B - 128) * tone * 0.1 * k + tone * 1.8 * k

    # 3. Blush on the cheekbones.
    blush = np.zeros((H, W), np.float32)
    for top, low in CHEEKS:
        c = (lm[top] * 0.6 + lm[low] * 0.4)
        cv2.ellipse(blush, tuple(np.round(c).astype(int)), (px(0.11), px(0.07)), 0, 0, 360, 1.0, -1)
    blush = cv2.GaussianBlur(blush, (0, 0), px(0.05)) * skin
    A += blush * 4 * k
    L += blush * 1.0 * k

    # 4. Eyes: clearer whites (less red/yellow, a bit brighter), brighter irises.
    eyes = np.zeros((H, W), np.float32)
    for idx in (LEFT_EYE, RIGHT_EYE):
        eyes = np.maximum(eyes, _poly_mask(rgb.shape, lm[idx], blur=px(0.004)))
    chroma = np.sqrt((A - 128) ** 2 + (B - 128) ** 2)
    if (eyes > 0.5).sum() > 20:
        lthr = float(np.percentile(L[eyes > 0.5], 55))
        white = eyes * np.clip((L - lthr) / 20, 0, 1) * np.clip((40 - chroma) / 20, 0, 1)
        A += (128 - A) * white * 0.5 * k
        B += (128 - B) * white * 0.5 * k
        L += white * 8 * k
        iris = eyes * (1 - white)
        L += iris * 3 * k

    # 5. Teeth: whiter where the mouth is open and the pixels are bright.
    mouth = _poly_mask(rgb.shape, lm[MOUTH_INNER], blur=px(0.004))
    if (mouth > 0.5).sum() > 20:
        mthr = float(np.percentile(L[mouth > 0.5], 50))
        teeth = mouth * np.clip((L - mthr) / 15, 0, 1) * np.clip((L - 90) / 30, 0, 1)
        B += (128 - B) * teeth * 0.6 * k
        A += (128 - A) * teeth * 0.3 * k
        L += teeth * 10 * k

    # 6. Lips: rosier and a little glossier.
    lips = _poly_mask(rgb.shape, lm[LIPS], blur=px(0.008)) * (1 - mouth)
    A += lips * 3.5 * k  # a touch rosier; no added gloss (it looked artificial)

    lab = np.stack([L, A, B], -1)
    out = _rgb(lab)

    # 7. Soft glow (bloom) over the face, screen-blended, never over the eye/lip detail.
    bloom = cv2.GaussianBlur(out, (0, 0), px(0.04))
    screen = 255 - (255 - out) * (255 - bloom) / 255
    g = (face * (1 - feats * 0.8) * 0.06 * k)[..., None]
    out = out * (1 - g) + screen * g

    # 8. Keep eyes, brows and lashes crisp after all the softening.
    sharp = out + (out - cv2.GaussianBlur(out, (0, 0), max(1.0, 0.005 * px(1)))) * 0.5 * k
    fm = (feats * face)[..., None]
    out = out * (1 - fm) + sharp * fm
    return np.clip(out, 0, 255)


def _masked_blur(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only averages pixels where weight > 0 (so hair and background don't bleed in)."""
    w = cv2.GaussianBlur(weight, (0, 0), sigma)[..., None]
    return cv2.GaussianBlur(img * weight[..., None], (0, 0), sigma) / np.maximum(w, 1e-4)


def face_lock(src: np.ndarray, src_lm: np.ndarray, out: np.ndarray, out_lm: np.ndarray | None,
              bangs: bool = False, enhance: float = 0.4, relight: float = 1.0) -> tuple[np.ndarray, bool]:
    """Blend the user's original face back onto the generated image. Returns (image, locked)."""
    if out_lm is None:
        return out, False
    M = align(src_lm, out_lm)
    if M is None:  # the model moved or reshaped the head a lot; a paste would look wrong
        return out, False
    g_out = geom(out_lm)

    H, W = out.shape[:2]
    warped = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    src_keep = cv2.warpAffine(keep_mask(src.shape, src_lm, bangs), M, (W, H), flags=cv2.INTER_LINEAR)
    region = cv2.min(src_keep, keep_mask(out.shape, out_lm, bangs))

    fw = g_out.width

    # Hair the AI drew in front of the face (braids falling over the cheeks or temples) must stay.
    # Only the outer band of the face is checked, never the eyes/nose/mouth core, and a pixel counts
    # as hair when it differs from the customer's skin far more than the overall lighting change.
    lab_s0 = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_o0 = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    inside0 = region > 128
    occluded = np.zeros(region.shape, np.float32)
    if inside0.sum() > 100:
        d = lab_o0 - lab_s0
        d -= np.median(d[inside0], axis=0)  # remove the global lighting/white-balance difference
        dist = np.sqrt((d[..., 0] * 0.8) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
        oval = out_lm[FACE_OVAL]
        c = oval.mean(axis=0)
        core = np.zeros(region.shape, np.uint8)
        cv2.fillPoly(core, [np.round(c + (oval - c) * 0.72).astype(np.int32)], 255)
        band = inside0 & (core == 0)
        hair = (band & (dist > 28)).astype(np.uint8) * 255
        hair = cv2.morphologyEx(hair, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop specks
        occluded = cv2.GaussianBlur(cv2.dilate(hair, np.ones((5, 5), np.uint8)), (0, 0), 2).astype(np.float32) / 255.0

    skin = (region > 0).astype(np.float32) * (occluded < 0.3)

    # Retouch the customer's own face (skin, tone, light only; nothing moves). enhance=0 turns it off.
    if enhance > 0:
        lm_w = src_lm @ M[:, :2].T + M[:, 2]
        warped = beautify(warped, lm_w, enhance)

    # Relight: keep the customer's fine detail (features, texture), take the overall lighting and
    # white balance from the new image so the face sits in the same light as the hair and background.
    # The shift is capped, so the customer's skin tone can only move a little.
    lab_s = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_o = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    sig = max(2.0, 0.06 * fw)
    low_s, low_o = _masked_blur(lab_s, skin, sig), _masked_blur(lab_o, skin, sig)
    delta = low_o - low_s
    delta[..., 0] = np.clip(delta[..., 0], -18 * relight, 18 * relight)    # lightness
    delta[..., 1:] = np.clip(delta[..., 1:], -14 * relight, 14 * relight)  # colour cast (camera white balance)
    face = lab_s + delta

    warped = cv2.cvtColor(np.clip(face, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Wide, soft edge so there is no visible outline.
    f = max(4, int(0.08 * fw))
    region = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=max(1, f // 4))
    alpha = cv2.GaussianBlur(region, (0, 0), sigmaX=f / 2).astype(np.float32) / 255.0
    alpha = (alpha * (1 - occluded))[..., None]

    blended = warped.astype(np.float32) * alpha + out.astype(np.float32) * (1 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8), True


def _skin_stats(rgb: np.ndarray, seg: np.ndarray):
    """Mean and spread of her face skin in Lab (lips, eyes and brows are not FACE_SKIN in the model)."""
    m = seg[..., FACE_SKIN] > 0.6
    if m.sum() < 500:
        return None
    lab = _lab(rgb)[m]
    return lab.mean(0), lab.std(0) + 1e-3


def skin_tone_gap(ref: np.ndarray, seg_ref: np.ndarray, out: np.ndarray, seg_out: np.ndarray):
    """How far the result's face skin moved from hers: (lightness change, colour change), Lab units.
    Positive lightness = the result is lighter."""
    a, b = _skin_stats(ref, seg_ref), _skin_stats(out, seg_out)
    if a is None or b is None:
        return None
    d = b[0] - a[0]
    return float(d[0] * 100 / 255), float(np.hypot(d[1], d[2]))


def keep_skin_tone(ref: np.ndarray, seg_ref: np.ndarray, out: np.ndarray, seg_out: np.ndarray,
                   strength: float = 1.0, exact: bool = False) -> np.ndarray:
    """Put her own skin tone back on the result's skin (face, neck, arms) when the AI lightened, darkened
    or tinted it. One smooth colour shift through a soft skin mask: nothing is pasted, nothing is moved,
    the AI's light and the texture it kept stay, so there is no edge or line anywhere."""
    a, b = _skin_stats(ref, seg_ref), _skin_stats(out, seg_out)
    if a is None or b is None or strength <= 0:
        return out
    (mr, sr), (mo, so) = a, b
    H, W = out.shape[:2]
    skin = np.clip((seg_out[..., FACE_SKIN] + seg_out[..., BODY_SKIN] - 0.3) / 0.4, 0, 1).astype(np.float32)
    # keep the wig and glasses untouched where they overlap skin, then feather widely
    skin *= 1 - np.clip((seg_out[..., HAIR] + seg_out[..., OTHERS] - 0.3) / 0.4, 0, 1)
    skin = cv2.GaussianBlur(skin, (0, 0), max(3.0, 0.012 * W)) * strength
    lab = _lab(out)
    # Her photo's own colour cast (blue office light, yellow bulbs) is not her skin: measure it on the
    # neutral greys/whites of each picture (ceiling, walls, backdrop) and leave it out of the target.
    def cast(img, seg):
        l = _lab(img)
        c = np.hypot(l[..., 1] - 128, l[..., 2] - 128)
        n = (c < 14) & (l[..., 0] > 90) & (seg[..., FACE_SKIN] + seg[..., BODY_SKIN] < 0.2)
        return (l[n][:, 1:].mean(0) - 128) if n.sum() > 800 else np.zeros(2, np.float32)
    # exact: her skin colour as photographed (her real face goes back on top, so they must match)
    want_ab = mr[1:] if exact else mr[1:] - (cast(ref, seg_ref) - cast(out, seg_out))
    # lightness: her mean, a contrast close to hers (capped so shading stays natural)
    ratio = float(np.clip(sr[0] / so[0], 0.85, 1.15))
    L = (lab[..., 0] - mo[0]) * ratio + mr[0]
    dL = np.clip(L - lab[..., 0], -40, 40)  # at most ~16 L* units either way
    # colour: half way to hers, so a photo taken in odd light never turns the result ashy or orange
    dab = np.clip(want_ab - mo[1:], -14, 14) if exact else np.clip(0.5 * (want_ab - mo[1:]), -6, 6)
    lab[..., 0] += dL * skin
    lab[..., 1] += dab[0] * skin
    lab[..., 2] += dab[1] * skin
    return _to_u8(_rgb(lab))


def garment_fit(src: np.ndarray, seg_src: np.ndarray, out: np.ndarray, seg_out: np.ndarray, M: np.ndarray | None):
    """Line her garment up with the result's. Returns (her photo moved onto the result, her garment mask
    there, overlap 0..1 of the two garments) or None when either picture shows no clothes."""
    H, W = out.shape[:2]
    if M is None:
        M = np.float32([[W / src.shape[1], 0, 0], [0, H / src.shape[0], 0]])
    warp = lambda x, f=cv2.INTER_LINEAR: cv2.warpAffine(x, M, (W, H), flags=f, borderMode=cv2.BORDER_REPLICATE)
    src_w = warp(src)
    c_src = warp(np.ascontiguousarray(seg_src[..., CLOTHES]), cv2.INTER_LINEAR)
    c_out = seg_out[..., CLOTHES]
    if (c_src > 0.5).sum() < 1500:
        return None  # she shows no clothes to keep
    if (c_out > 0.5).sum() < 1500:
        return src_w, c_src, 0.0  # her clothes are gone from the result
    # The body can sit a little differently from the face: line the garments up with each other
    (dx, dy), _ = cv2.phaseCorrelate(c_src.astype(np.float64), c_out.astype(np.float64))
    if abs(dx) < 0.08 * W and abs(dy) < 0.08 * H:
        T = np.float32([[1, 0, dx], [0, 1, dy]])
        src_w = cv2.warpAffine(src_w, T, (W, H), borderMode=cv2.BORDER_REPLICATE)
        c_src = cv2.warpAffine(c_src, T, (W, H))
    # compare only where the new hair doesn't cover (long hair hides part of any garment)
    free = seg_out[..., HAIR] < 0.35
    # only inside her photo: below a close selfie the AI simply continues her top, that is not a change
    free &= cv2.warpAffine(np.ones(src.shape[:2], np.float32), M, (W, H)) > 0.99
    a, b = (c_src > 0.5) & free, (c_out > 0.5) & free
    iou = float((a & b).sum()) / float((a | b).sum() + 1)
    return src_w, c_src, iou


def restore_clothes(src: np.ndarray, seg_src: np.ndarray, out: np.ndarray, seg_out: np.ndarray,
                    M: np.ndarray | None) -> tuple[np.ndarray, str]:
    """Her clothes, exactly as photographed, on the result.

    1. The result's whole garment is recoloured to her garment's colours (so every bit the AI drew,
       including between braids and at the edges, is already the right colour: no odd rims).
    2. Her real garment pixels then go on top wherever the result has clothes and no hair.
    Only the CLOTHES class is used (never background or accessories), hair always stays the AI's,
    and nothing happens if her garment can't be lined up with the result's.
    Returns (image, note for the log)."""
    H, W = out.shape[:2]
    fit = garment_fit(src, seg_src, out, seg_out, M)
    if fit is None:
        return out, "no clothes to keep"
    src_w, c_src, iou = fit
    c_out = seg_out[..., CLOTHES]
    a, b = c_src > 0.5, c_out > 0.5
    if iou < 0.3:
        return out, f"garment shape too different to line up (overlap {iou:.2f})"
    colour_only = iou < 0.5  # lined up too loosely to place her pixels, but her colours still apply

    hair = np.clip((seg_out[..., HAIR] - 0.35) / 0.3, 0, 1).astype(np.float32)
    # 1. recolour the AI's garment to hers (mean and spread per Lab channel)
    lo, ls = _lab(out), _lab(src_w)
    ref = ls[a & ~(hair > 0.5)] if (a & ~(hair > 0.5)).sum() > 800 else ls[a]
    cur = lo[b & ~(hair > 0.5)] if (b & ~(hair > 0.5)).sum() > 800 else lo[b]
    mr, sr, mo, so = ref.mean(0), ref.std(0) + 1e-3, cur.mean(0), cur.std(0) + 1e-3
    k = np.clip(sr / so, 0.6, 1.6)
    moved = (lo - mo) * k + mr
    w1 = cv2.GaussianBlur(np.clip((c_out - 0.3) / 0.4, 0, 1).astype(np.float32), (0, 0), 1.5)
    w1 = (w1 * (1 - hair))[..., None]
    lo = lo * (1 - w1) + moved * w1
    res = _rgb(lo)
    if colour_only:
        return _to_u8(res), f"recoloured only (overlap {iou:.2f})"
    # 2. her real garment where both pictures have it and no hair lies on it
    grow = cv2.dilate(b.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    keep = (a & grow).astype(np.float32)
    keep = cv2.erode(keep, np.ones((3, 3), np.uint8))
    hair_block = cv2.dilate((seg_out[..., HAIR] > 0.35).astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    keep = cv2.GaussianBlur(keep * (1 - hair_block), (0, 0), 1.2)[..., None]
    res = src_w.astype(np.float32) * keep + res * (1 - keep)
    return _to_u8(res), f"kept (overlap {iou:.2f}, {float(keep.mean()) * 100:.0f}% of picture)"


def keep_body(src: np.ndarray, seg_src: np.ndarray, out: np.ndarray, seg_out: np.ndarray,
              M: np.ndarray, inset: int = 2, follow: bool = False,
              valid: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Her own body and clothes, always, by construction (whatever the AI drew there).

    The AI's picture supplies only the new hair, the head and the backdrop. Her real clothes, arms,
    hands and neck from her photo are placed exactly where they belong relative to her face (the face
    was already put back the same way), under the new hair. Anything the AI invented outside her real
    body (a new shirt, extra shoulders) becomes backdrop. Returns (image, note for the log)."""
    H, W = out.shape[:2]
    warp = lambda x, f=cv2.INTER_LINEAR, b=cv2.BORDER_CONSTANT: cv2.warpAffine(x, M, (W, H), flags=f, borderMode=b)
    src_w = warp(src, b=cv2.BORDER_REFLECT).astype(np.float32)
    inside = cv2.erode((warp(np.ones(src.shape[:2], np.float32)) > 0.99).astype(np.uint8),
                       np.ones((5, 5), np.uint8)).astype(np.float32)  # pixels that really come from her photo
    seg_w = np.stack([warp(np.ascontiguousarray(seg_src[..., i])) for i in range(seg_src.shape[-1])], -1)
    body = np.clip(seg_w[..., CLOTHES] + seg_w[..., BODY_SKIN], 0, 1)
    near = cv2.dilate((body > 0.5).astype(np.uint8), np.ones((int(0.05 * W) | 1,) * 2, np.uint8)).astype(np.float32)
    body = np.clip(body + seg_w[..., OTHERS] * near, 0, 1)  # bags, straps, jewellery on her
    # Light or shiny garments (a white top under a jacket, satin) are often missed by the segmenter and
    # come out as holes in her body. Close small gaps and fill enclosed holes, so they stay hers instead
    # of turning into backdrop. Large enclosed areas (real background between arm and body) stay open.
    bb = (body > 0.5).astype(np.uint8)
    k = int(0.02 * W) | 1
    bb = cv2.morphologyEx(bb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inv = (1 - bb).astype(np.uint8)
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, w_, h_, area = stats[i]
        touches = x == 0 or y == 0 or x + w_ >= W or y + h_ >= H
        if not touches and area < 0.04 * H * W:
            bb[lab_ == i] = 1
    body = np.maximum(body, bb.astype(np.float32) * 0.99)
    if (body > 0.5).sum() < 1500:
        return out, "no body to keep"
    old_hair = seg_w[..., HAIR]
    new_hair = seg_out[..., HAIR]
    res = _lab(out)

    # 1. The AI's clothes take her garment's colours (for any bit of them that stays visible).
    c_src, c_out = seg_w[..., CLOTHES] > 0.5, (seg_out[..., CLOTHES] > 0.5) & (new_hair < 0.35)
    if c_src.sum() > 800 and c_out.sum() > 800:
        ls = _lab(src_w)
        mr, sr = ls[c_src].mean(0), ls[c_src].std(0) + 1e-3
        mo, so = res[c_out].mean(0), res[c_out].std(0) + 1e-3
        moved = (res - mo) * np.clip(sr / so, 0.6, 1.6) + mr
        w = cv2.GaussianBlur(np.clip((seg_out[..., CLOTHES] - 0.3) / 0.4, 0, 1).astype(np.float32), (0, 0), 1.5)
        w = (w * (1 - np.clip((new_hair - 0.35) / 0.3, 0, 1)))[..., None]
        res = res * (1 - w) + moved * w
    res = _rgb(res)

    # 2. Body the AI invented where her photo had only background: plain backdrop instead.
    #    (skipped on the blended pass: there the AI already finished those edges)
    ai_body = np.clip(seg_out[..., CLOTHES] + seg_out[..., BODY_SKIN] + seg_out[..., OTHERS], 0, 1)
    was_bg = (body < 0.3) & (old_hair < 0.3) & (seg_w[..., FACE_SKIN] < 0.3)
    # Only where her photo really shows background. Outside her photo (below its bottom edge, the room
    # added above/beside a close selfie) the AI's drawing is the only picture there is, so it stays.
    cover = np.ones(src.shape[:2], np.float32) if valid is None else valid.astype(np.float32)
    cover = warp(cover) > 0.99
    cover = cv2.erode(cover.astype(np.uint8), np.ones((int(0.03 * W) | 1,) * 2, np.uint8)).astype(bool)
    was_bg &= cover
    invented = ((ai_body > 0.5) & was_bg & (new_hair < 0.35)).astype(np.uint8)
    invented = cv2.morphologyEx(invented, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # follow = the AI re-posed her: keep its pose (her arms where it put them), so nothing of her old
    # pose is pasted where the result has no body (no floating sleeves or hands)
    # Only on the first assembly. After the AI blend pass its edges, shadows and the surface under her
    # arms are what make the picture whole, so they stay (stray pieces are caught by tidy_body).
    if invented.any() and not follow and inset <= 2:
        bgm = cv2.erode((seg_out[..., BG] > 0.7).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
        sig = max(8.0, 0.08 * W)
        num = cv2.GaussianBlur(out.astype(np.float32) * bgm[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(bgm, (0, 0), sig)[..., None]
        fallback = np.median(out[bgm > 0], axis=0).astype(np.float32) if bgm.sum() > 500 else np.float32([225, 225, 225])
        plate = np.where(den > 0.05, num / np.maximum(den, 1e-3), fallback)
        a = cv2.GaussianBlur(cv2.dilate(invented, np.ones((11, 11), np.uint8)).astype(np.float32), (0, 0), 3.0)
        a = (a * (1 - np.clip((new_hair - 0.2) / 0.3, 0, 1)))[..., None]  # never wash out the new hair
        res = res * (1 - a) + plate * a

    # 3. Her real body on top, under the new hair. Edge pulled in 1 px so none of her old room shows.
    # (inset > 1: only the inside of her clothes/body, the edges stay as the AI blended them)
    e = 2 * inset + 1
    keep = cv2.erode((body > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))).astype(np.float32)
    g = 2 * max(2, inset // 2) + 1
    block = cv2.dilate((new_hair > 0.35).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    if follow:
        there = cv2.dilate((ai_body > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        keep *= there
    keep = keep * inside
    if valid is not None:  # nor from the soft room added around a close selfie
        keep *= cv2.erode((warp(valid.astype(np.float32)) > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8))
    keep = cv2.GaussianBlur(keep * (1 - block), (0, 0), max(1.0, inset / 2.5))[..., None]
    res = src_w * keep + res * (1 - keep)
    return _to_u8(res), f"kept ({float(keep.mean()) * 100:.0f}% of picture, {int(invented.sum())} invented px removed)"


def tidy_body(out: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, int]:
    """Last safety net on the finished picture: any piece of clothing or skin that floats on its own,
    not joined to her body (a stray sleeve, a hand left from an old pose, a patch at the edge), is
    replaced with the backdrop. Her body, face and the hair are never touched. Returns (image, pieces)."""
    H, W = out.shape[:2]
    person = np.clip(seg[..., CLOTHES] + seg[..., BODY_SKIN] + seg[..., FACE_SKIN] + seg[..., HAIR]
                     + seg[..., OTHERS], 0, 1) > 0.5
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(person.astype(np.uint8), connectivity=8)
    if n <= 2:
        return out, 0
    face = seg[..., FACE_SKIN] > 0.5
    main = max(range(1, n), key=lambda i: (face[lab_ == i].sum(), stats[i][4]))  # the piece with her face
    stray = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        if i != main and stats[i][4] < 0.015 * H * W:  # specks and slivers, never a real part of the picture
            stray[lab_ == i] = 1
    stray &= ((seg[..., HAIR] < 0.35)).astype(np.uint8)  # loose curls are allowed to float
    if not stray.any():
        return out, 0
    bgm = cv2.erode((seg[..., BG] > 0.7).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    sig = max(8.0, 0.08 * W)
    num = cv2.GaussianBlur(out.astype(np.float32) * bgm[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(bgm, (0, 0), sig)[..., None]
    fallback = np.median(out[bgm > 0], axis=0).astype(np.float32) if bgm.sum() > 500 else np.float32([225, 225, 225])
    plate = np.where(den > 0.05, num / np.maximum(den, 1e-3), fallback)
    a = cv2.GaussianBlur(cv2.dilate(stray, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 2.5)
    a = (a * (1 - np.clip((seg[..., HAIR] - 0.2) / 0.3, 0, 1)))[..., None]
    res = out.astype(np.float32) * (1 - a) + plate * a
    return _to_u8(res), int(n - 2)


def no_new_hands(out: np.ndarray, seg_out: np.ndarray, src_skin_w: np.ndarray,
                 inside: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Remove skin the AI drew where her photo has no skin (a second pair of hands, a new arm).
    Her own hands and arms, wherever they are visible, stay; hands hidden behind the new hair stay hidden.
    src_skin_w: her body-skin probability moved onto the result. Returns (image, removed px)."""
    H, W = out.shape[:2]
    mine = cv2.dilate((src_skin_w > 0.3).astype(np.uint8), np.ones((int(0.02 * W) | 1,) * 2, np.uint8))
    new = ((seg_out[..., BODY_SKIN] > 0.5) & (mine == 0) & (seg_out[..., HAIR] < 0.35)).astype(np.uint8)
    if inside is not None:  # only within her photo: below/around it the AI's picture is all there is
        new &= cv2.erode(inside.astype(np.uint8), np.ones((int(0.03 * W) | 1,) * 2, np.uint8))
    new = cv2.morphologyEx(new, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # only real pieces (a hand is not a few stray pixels)
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(new, connectivity=8)
    keep = np.zeros_like(new)
    for i in range(1, n):
        if stats[i][4] > 0.002 * H * W:
            keep[lab_ == i] = 1
    if not keep.any():
        return out, 0
    # removed smoothly (inpainting smears streaks across the picture)
    m = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # fill from what surrounds it (her top, or the backdrop), never from hair: a smooth patch, no streaks
    ok = ((m == 0) & (seg_out[..., HAIR] < 0.35)).astype(np.float32)
    plate = out.astype(np.float32)
    for sig in (0.02 * W, 0.05 * W, 0.12 * W):
        num = cv2.GaussianBlur(out.astype(np.float32) * ok[..., None], (0, 0), max(3.0, sig))
        den = cv2.GaussianBlur(ok, (0, 0), max(3.0, sig))[..., None]
        fill = num / np.maximum(den, 1e-3)
        plate = np.where((den > 0.15) & (m[..., None] > 0) & (plate == out), fill, plate)
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3.0)
    a = (a * (1 - np.clip((seg_out[..., HAIR] - 0.2) / 0.3, 0, 1)))[..., None]
    res = out.astype(np.float32) * (1 - a) + plate * a
    return _to_u8(res), int(keep.sum())
