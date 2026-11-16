import tensorflow as tf
import numpy as np
import matplotlib
from PIL import Image
from utils.logger import logger
from utils.model_architecture import GRADCAM_LAYER


def find_target_layer(model):
    """Last spatial feature layer: the named Grad-CAM target, else the last DWSC block (models saved before the named target existed)."""
    names = [layer.name for layer in model.layers]
    if GRADCAM_LAYER in names:
        return GRADCAM_LAYER
    return next((n for n in reversed(names) if "depthwise_separable_conv" in n), None)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, explain_malignant=None):
    """
    Grad-CAM for a single-sigmoid binary classifier.

    The previous version took `argmax` over the one-element sigmoid output, which is
    always index 0, so every heatmap explained the malignant score regardless of the
    prediction. Here we differentiate the logit: +logit explains "malignant",
    -logit explains "benign". By default the predicted class is explained.
    """
    try:
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        logger.error(f"Error creating Grad-CAM model: {e}")
        return None

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        if isinstance(preds, list):
            preds = preds[0]
        p = tf.clip_by_value(preds[:, 0], 1e-6, 1 - 1e-6)
        logit = tf.math.log(p / (1 - p))
        if explain_malignant is None:
            explain_malignant = bool(p[0] >= 0.5)
        target = logit if explain_malignant else -logit

    grads = tape.gradient(target, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    peak = tf.reduce_max(heatmap)
    if peak <= 0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    return (heatmap / peak).numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """
    Superimposes the heatmap on an image.
    img: PIL Image or uint8 numpy array; must be the same view the model saw
         (the preprocessed 224x224 frame) so the overlay is spatially aligned.
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = np.asarray(img.convert("RGB"), dtype=np.float32)

    jet = matplotlib.colormaps["jet"](np.arange(256))[:, :3]
    colored = jet[np.uint8(255 * heatmap)] * 255
    colored = np.asarray(
        Image.fromarray(colored.astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.BILINEAR),
        dtype=np.float32,
    )
    blended = np.clip(colored * alpha + img * (1 - alpha), 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
