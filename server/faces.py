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
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def landmarks(self, rgb: np.ndarray) -> np.ndarray | None:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect(img)
        if not res.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)


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
    return zone


def head_area(src_shape, src_lm: np.ndarray) -> np.ndarray:
    """float 0/1 ellipse over the top of the head: crown, hairline, temples and ears."""
    H, W = src_shape[:2]
    g = geom(src_lm)
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, tuple(np.round(src_lm[FOREHEAD] + g.up * 0.1 * g.height).astype(int)),
                (int(1.05 * g.width), int(0.85 * g.height)), 0, 0, 360, 1.0, -1)
    return m


def transfer_hair(src: np.ndarray, src_lm: np.ndarray, out: np.ndarray, out_lm: np.ndarray | None,
                  zone: np.ndarray, seg_src: np.ndarray | None = None, seg_out: np.ndarray | None = None,
                  enhance: float = 0.75, background: str = "studio") -> np.ndarray:
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
    base = beautify(src, src_lm, enhance) if enhance > 0 else src.copy()
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
    old = (seg_src[..., HAIR] > 0.4).astype(np.uint8) * 255 if seg_src is not None else np.zeros((H, W), np.uint8)
    old[_poly_mask(src.shape, src_lm[FACE_OVAL]) > 0.5] = 0  # never touch the face
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
    hair_src = np.clip((seg_src[..., HAIR] - 0.25) / 0.4, 0, 1) if seg_src is not None else np.zeros((H, W), np.float32)
    hair_src = cv2.dilate(hair_src, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, int(0.01 * W)) + 1,) * 2))
    not_bg_out = 1 - np.clip(to_src(np.ascontiguousarray(seg_out[..., BG])), 0, 1)
    flush = head * np.maximum(hair_src, p) * np.clip((not_bg_out - 0.3) / 0.4, 0, 1)
    a = np.maximum(a, cv2.GaussianBlur(flush, (0, 0), 1.0))

    final = base.astype(np.float32) * (1 - a[..., None]) + out_s * a[..., None]

    if background in ("studio", "white") and seg_src is not None:
        # Keep the person (their clothes, skin, face and accessories) plus the new hair; replace the rest.
        # Everything that isn't background or their old hair counts as the person, so unusual garments
        # and accessories are always kept rather than cut away.
        person = np.clip(1 - seg_src[..., BG] - seg_src[..., HAIR], 0, 1)
        person = np.maximum(person, _poly_mask(src.shape, src_lm[FACE_OVAL]))  # the face is always kept
        person = np.clip((person - 0.3) / 0.4, 0, 1)                          # crisper edge, no room halo
        person = cv2.GaussianBlur(person, (0, 0), 1.0)
        alpha = np.maximum(person, a)[..., None]
        final = final * alpha + backdrop(H, W, background) * (1 - alpha)
    return np.clip(final, 0, 255).astype(np.uint8)


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


def beautify(rgb: np.ndarray, lm: np.ndarray, strength: float = 0.75) -> np.ndarray:
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

    # 1. Skin smoothing: remove blemishes and unevenness (mid frequencies), keep fine texture.
    base = cv2.bilateralFilter(rgb, d=0, sigmaColor=28, sigmaSpace=px(0.02))
    base = cv2.bilateralFilter(base, d=0, sigmaColor=22, sigmaSpace=px(0.015)).astype(np.float32)
    fine = img - cv2.GaussianBlur(img, (0, 0), max(0.8, 0.004 * fw))
    smooth = cv2.GaussianBlur(base, (0, 0), max(0.6, 0.002 * fw)) + fine * 0.55
    a = (skin * 0.85 * k)[..., None]
    img = img * (1 - a) + smooth * a

    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
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
        L += face * 4.0 * k + under * 6.0 * k

    lab = np.stack([L, A, B], -1)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # 5. Crisper eyes, brows and lips; a touch of lip colour.
    sharp = out + (out - cv2.GaussianBlur(out, (0, 0), max(1.0, 0.006 * fw))) * 0.6 * k
    fm = (feats * face)[..., None]
    out = out * (1 - fm) + sharp * fm
    lips = _poly_mask(rgb.shape, lm[LIPS], blur=px(0.008))
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1 + lips * 0.15 * k
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def _masked_blur(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only averages pixels where weight > 0 (so hair and background don't bleed in)."""
    w = cv2.GaussianBlur(weight, (0, 0), sigma)[..., None]
    return cv2.GaussianBlur(img * weight[..., None], (0, 0), sigma) / np.maximum(w, 1e-4)


def face_lock(src: np.ndarray, src_lm: np.ndarray, out: np.ndarray, out_lm: np.ndarray | None,
              bangs: bool = False, enhance: float = 0.4) -> tuple[np.ndarray, bool]:
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
    delta[..., 0] = np.clip(delta[..., 0], -18, 18)    # lightness
    delta[..., 1:] = np.clip(delta[..., 1:], -14, 14)  # colour cast (camera white balance)
    face = lab_s + delta

    warped = cv2.cvtColor(np.clip(face, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Wide, soft edge so there is no visible outline.
    f = max(4, int(0.08 * fw))
    region = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=max(1, f // 4))
    alpha = cv2.GaussianBlur(region, (0, 0), sigmaX=f / 2).astype(np.float32) / 255.0
    alpha = (alpha * (1 - occluded))[..., None]

    blended = warped.astype(np.float32) * alpha + out.astype(np.float32) * (1 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8), True