import cv2
import numpy as np
from PIL import Image
from utils.logger import logger

IMG_SIZE = 224

# Pixels darker than this are treated as background / scanner frame when cropping.
_BORDER_THRESHOLD = 12
# Max-min channel spread above which a pixel is a coloured overlay (B-mode tissue is grey).
_OVERLAY_SATURATION = 40


def remove_color_overlays(image):
    """
    Converts to grayscale and inpaints coloured scanner overlays (text, labels, colour markers).

    B-mode ultrasound is intrinsically grayscale, so a saturated pixel can only be a
    burned-in annotation, never tissue. This is deliberately NOT done by brightness:
    bright echogenic foci (microcalcifications) are a key malignancy feature and must survive.
    """
    if image.mode in ("L", "I", "I;16"):
        return np.asarray(image.convert("L"))
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    gray = np.asarray(image.convert("L"))
    mask = (rgb.max(axis=-1) - rgb.min(axis=-1)) > _OVERLAY_SATURATION
    if not mask.any():
        return gray
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


def content_bounds(gray):
    """Row/column bounds of the ultrasound field, excluding the dark scanner frame."""
    rows = np.where(gray.mean(axis=1) > _BORDER_THRESHOLD)[0]
    cols = np.where(gray.mean(axis=0) > _BORDER_THRESHOLD)[0]
    if len(rows) < 16 or len(cols) < 16:
        return 0, gray.shape[0], 0, gray.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def crop_to_content(gray):
    """Crops the black scanner frame around the ultrasound sector."""
    r0, r1, c0, c1 = content_bounds(gray)
    return gray[r0:r1, c0:c1]


def letterbox(gray, size=IMG_SIZE, return_transform=False):
    """Resizes keeping aspect ratio and pads with black to a square canvas."""
    h, w = gray.shape
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = np.asarray(Image.fromarray(gray).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    if return_transform:
        return canvas, (scale, left, top)
    return canvas


def prepare_ultrasound(image, size=IMG_SIZE, return_transform=False):
    """
    Deterministic preprocessing shared by training and inference.

    1. Grayscale, with coloured scanner overlays inpainted (B-mode carries no colour).
    2. Crop the dark scanner frame.
    3. Letterbox-resize to size x size so nodule geometry (taller-than-wide is a
       TI-RADS malignancy feature) is not distorted by a naive stretch.
    Returns a uint8 array of shape (size, size). With `return_transform`, also returns
    a function mapping original-image (x, y) coordinates into the output frame.
    """
    gray = remove_color_overlays(image)
    r0, r1, c0, c1 = content_bounds(gray)
    out, (scale, left, top) = letterbox(gray[r0:r1, c0:c1], size, return_transform=True)
    if not return_transform:
        return out
    return out, lambda x, y: ((x - c0) * scale + left, (y - r0) * scale + top)


def to_model_input(gray_uint8):
    """uint8 (H, W) or (N, H, W) -> float32 3-channel tensor in [0, 1]."""
    x = gray_uint8.astype("float32") / 255.0
    return np.repeat(x[..., None], 3, axis=-1)


def preprocess_image(image):
    """
    Preprocesses a PIL image for inference. Returns a (1, 224, 224, 3) float32 batch.
    """
    logger.info(f"Preprocessing image of size: {image.size} and mode: {image.mode}")
    return to_model_input(prepare_ultrasound(image))[None, ...]
