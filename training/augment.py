"""
Ultrasound-aware augmentation, applied to the TRAIN split only, on the fly.

Each transform mimics a real source of acquisition variability:
- horizontal flip       left/right lobe and probe orientation are arbitrary
- small rotation/zoom/  probe angle, depth setting and footprint position
  translation
- gain / contrast /     operator-dependent overall gain and TGC settings
  gamma
- multiplicative        speckle, the characteristic ultrasound noise
  speckle noise
- random erasing        occlusion; also weakens reliance on burned-in calipers

No vertical flip: depth is meaningful in B-mode (skin at the top, acoustic
shadowing below), and flipping it creates anatomically impossible images.
"""
import tensorflow as tf
from tensorflow.keras import layers

_geometric = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(10 / 360, fill_mode="constant"),
    layers.RandomZoom((-0.1, 0.15), fill_mode="constant"),
    layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
], name="geometric_aug")


# Intensity ranges per strength: (gain, contrast, log-gamma, max speckle sigma, erase probability)
STRENGTHS = {
    "standard": (0.2, 0.25, 0.25, 0.12, 0.3),
    "light": (0.1, 0.12, 0.12, 0.06, 0.15),
}


def _intensity(x, gain_r, contrast_r, gamma_r, sigma_max):
    gain = tf.random.uniform([], 1 - gain_r, 1 + gain_r)
    contrast = tf.random.uniform([], 1 - contrast_r, 1 + contrast_r)
    gamma = tf.exp(tf.random.uniform([], -gamma_r, gamma_r))
    mean = tf.reduce_mean(x)
    x = (x - mean) * contrast + mean
    x = tf.clip_by_value(x * gain, 0.0, 1.0) ** gamma
    sigma = tf.random.uniform([], 0.0, sigma_max)
    x = x * (1.0 + sigma * tf.random.normal(tf.shape(x)))
    return tf.clip_by_value(x, 0.0, 1.0)


def _random_erase(x, p=0.3):
    if tf.random.uniform([]) > p:
        return x
    h, w = tf.shape(x)[0], tf.shape(x)[1]
    eh = tf.cast(tf.cast(h, tf.float32) * tf.random.uniform([], 0.08, 0.25), tf.int32)
    ew = tf.cast(tf.cast(w, tf.float32) * tf.random.uniform([], 0.08, 0.25), tf.int32)
    y0 = tf.random.uniform([], 0, h - eh, dtype=tf.int32)
    x0 = tf.random.uniform([], 0, w - ew, dtype=tf.int32)
    mask = tf.pad(tf.ones([eh, ew, 1]), [[y0, h - y0 - eh], [x0, w - x0 - ew], [0, 0]])
    return x * (1.0 - mask)


def augment(image, strength="standard"):
    """image: float32 (H, W, 1) in [0, 1] -> augmented (H, W, 1)."""
    gain_r, contrast_r, gamma_r, sigma_max, erase_p = STRENGTHS[strength]
    image = _geometric(image[None, ...], training=True)[0]
    image = _intensity(image, gain_r, contrast_r, gamma_r, sigma_max)
    return _random_erase(image, erase_p)
