"""
Trains one architecture with one seed on the leakage-safe split.

    python -m training.train --model fibonacci_net --seed 0 [--dataset tn5000]

Outputs (runs/<dataset>/<model>/seed<k>/): model.keras, history.json, history.csv,
preds.npz (validation, test and external-test probabilities, used by training.evaluate).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from training.augment import augment
from training.dataset import load_cache, paths_for
from utils.model_architecture import MODEL_BUILDERS

RUNS_DIR = Path("runs")


class KeepBest(tf.keras.callbacks.Callback):
    """Tracks the best val AUROC epoch and restores its weights when training ends."""

    def __init__(self):
        super().__init__()
        self.best, self.weights = -1.0, None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_auc"] > self.best:
            self.best, self.weights = logs["val_auc"], self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.weights is not None:
            self.model.set_weights(self.weights)


def provenance(started, dataset):
    """Enough context to tie a result back to the exact code, data split and environment."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "training", "utils")),
        "splits_sha256": hashlib.sha256(paths_for(dataset)["splits"].read_bytes()).hexdigest()[:16],
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
    }


def make_ds(images, labels, training, batch_size, seed, aug="standard"):
    x = images[..., None].astype(np.float32) / 255.0
    ds = tf.data.Dataset.from_tensor_slices((x, labels))
    if training:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda im, y: (augment(im, aug), y), num_parallel_calls=tf.data.AUTOTUNE)
    # Grayscale ultrasound replicated to 3 channels to match the model input.
    ds = ds.map(lambda im, y: (tf.repeat(im, 3, axis=-1), y))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train(model_name, seed, epochs, batch_size, lr, patience, schedule="plateau", warmup=5, aug="standard",
          run_name=None, dataset="tn5000"):
    tf.keras.utils.set_random_seed(seed)
    data = load_cache(dataset)
    (xtr, ytr, _), (xva, yva, fva), (xte, yte, fte) = data["train"], data["val"], data["test"]

    # Class-weighted loss instead of oversampling: no duplicated images, no leakage.
    pos = ytr.mean()
    class_weight = {0: 0.5 / (1 - pos), 1: 0.5 / pos}

    model = MODEL_BUILDERS[model_name](input_shape=(224, 224, 3), num_classes=1)
    steps = int(np.ceil(len(xtr) / batch_size))
    if schedule == "cosine":
        # Linear warmup then cosine decay over the full budget; no early stopping.
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            0.0, decay_steps=(epochs - warmup) * steps, warmup_target=lr, warmup_steps=warmup * steps, alpha=0.01)
    else:
        learning_rate = lr
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )

    out = RUNS_DIR / dataset / (run_name or model_name) / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    callbacks = [KeepBest(), tf.keras.callbacks.CSVLogger(str(out / "history.csv"))]
    if schedule == "plateau":
        callbacks += [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                                                 patience=max(2, patience // 3), min_lr=1e-6),
        ]

    t0 = time.time()
    hist = model.fit(
        make_ds(xtr, ytr, True, batch_size, seed, aug),
        validation_data=make_ds(xva, yva, False, batch_size, seed),
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.time() - t0

    p_val = model.predict(make_ds(xva, yva, False, 64, seed), verbose=0).ravel()
    p_test = model.predict(make_ds(xte, yte, False, 64, seed), verbose=0).ravel()
    preds = dict(p_val=p_val, y_val=yva, p_test=p_test, y_test=yte, files_val=fva, files_test=fte)
    if "external" in data:
        xex, yex, fex = data["external"]
        preds.update(p_ext=model.predict(make_ds(xex, yex, False, 64, seed), verbose=0).ravel(),
                     y_ext=yex, files_ext=fex)
    np.savez(out / "preds.npz", **preds)
    model.save(out / "model.keras")
    meta = {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmax(hist.history["val_auc"])) + 1,
        "best_val_auc": float(max(hist.history["val_auc"])),
        "provenance": provenance(t0, dataset),
        "train_seconds": round(train_seconds, 1),
        "params": int(model.count_params()),
        "recipe": {"lr": lr, "schedule": schedule, "warmup": warmup, "aug": aug, "batch_size": batch_size,
                   "epochs": epochs},
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
    }
    (out / "history.json").write_text(json.dumps(meta, indent=2))
    print(f"[{model_name} seed{seed}] best val AUC {meta['best_val_auc']:.4f} "
          f"in {meta['epochs_run']} epochs ({train_seconds / 60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BUILDERS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10, help="early stopping (plateau schedule only)")
    parser.add_argument("--schedule", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--warmup", type=int, default=5, help="warmup epochs (cosine schedule only)")
    parser.add_argument("--aug", choices=["standard", "light"], default="standard")
    parser.add_argument("--run-name", help="output folder under runs/<dataset>/ (default: the model name)")
    parser.add_argument("--dataset", choices=["tn5000", "kaggle"], default="tn5000")
    args = parser.parse_args()
    train(args.model, args.seed, args.epochs, args.batch_size, args.lr, args.patience,
          args.schedule, args.warmup, args.aug, args.run_name, args.dataset)


if __name__ == "__main__":
    main()
