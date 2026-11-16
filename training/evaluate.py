"""
Aggregates all runs into reports/benchmark.json and exports the deployment model.

Data: TN5000 (group-safe official split) for development/test; the Kaggle set as an
EXTERNAL test that is never trained on, calibrated on or thresholded on.

Clinical evaluation protocol (all decisions fixed on VALIDATION, reported on TEST):
- Discrimination: AUROC / AUPRC with 95% bootstrap CIs.
- Calibration: Platt scaling (logistic recalibration of the logit) fitted on val; Brier score and
  ECE on test. Platt rather than temperature scaling because the class-weighted loss shifts the
  model's prior towards 50/50 (mean val prediction 0.66 vs 0.755 prevalence) and only Platt's
  intercept can correct that; it also had lower log-loss and ECE in 5-fold CV on val.
- Operating points chosen on val:
    * "balanced"  - Youden's J
    * "rule_in"   - lowest threshold with val specificity >= 90%; above it: high suspicion
    * "rule_out"  - highest threshold with val sensitivity >= 90%, the regime a
                    triage tool for FNA referral would run in (missing cancer is costlier).
  Sensitivity, specificity, PPV, NPV reported on test at those fixed thresholds.
- FibonacciNet vs baseline: paired DeLong test and a bootstrap non-inferiority
  test on the AUROC difference (margin 0.03).
- External validity: the TN5000-fitted calibration and thresholds are applied unchanged
  to the external set (different scanners, population and malignancy prevalence).
- Localization: Grad-CAM "pointing game" against the annotated nodule boxes - does the
  model look at the nodule, or at calipers/text/body marks?
- Efficiency: parameters, FLOPs, model file size, CPU batch-1 latency.
"""
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

RUNS_DIR = Path("runs/tn5000")
SWEEP_DIR = Path("runs/kaggle")
REPORT_PATH = Path("reports/benchmark.json")
SWEEP_PATH = Path("reports/sweep.json")
ARCHIVE_DIR = Path("reports/runs")
LOG_ARCHIVE = Path("reports/logs")
DEPLOY_DIR = Path("models")
PRIMARY, BASELINE = "fibonacci_net", "baseline_cnn"
EXPECTED_SEEDS = 3
NI_MARGIN = 0.03
RULE_OUT_SENS = 0.90
RULE_IN_SPEC = 0.90
N_BOOT = 2000
rng = np.random.default_rng(0)

DISPLAY = {
    "fibonacci_net": "FibonacciNet",
    "baseline_cnn": "Standard CNN baseline",
}


# ---------------------------------------------------------------- statistics
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    """Logistic recalibration sigmoid(a * logit(p) + b), fitted by minimising log-loss."""
    from scipy.optimize import minimize

    z, s = logit(p), 2 * y - 1
    res = minimize(lambda w: np.mean(np.logaddexp(0, -s * (w[0] * z + w[1]))), x0=[1.0, 0.0])
    return {"method": "platt", "a": float(res.x[0]), "b": float(res.x[1])}


def apply_calibration(p, cal):
    return 1 / (1 + np.exp(-(cal["a"] * logit(p) + cal["b"])))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
            out.append({"bin": b, "pred": float(p[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
    return float(total), out


def confusion_metrics(p, y, thr):
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    return {"threshold": float(thr), "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "accuracy": div(tp + tn, len(y)),
            "f1": div(2 * tp, 2 * tp + fp + fn), "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def pick_thresholds(p_val, y_val):
    fpr, tpr, thr = roc_curve(y_val, p_val)
    youden = thr[int(np.argmax(tpr - fpr))]
    ok = np.where(tpr >= RULE_OUT_SENS)[0]
    rule_out = min(thr[ok[0]] if len(ok) else thr[-1], 1.0)
    # Last ROC point still meeting the specificity target = lowest qualifying threshold.
    spec_ok = np.where(1 - fpr >= RULE_IN_SPEC)[0]
    rule_in = min(thr[spec_ok[-1]] if len(spec_ok) else 1.0, 1.0)
    # Bands: < rule_out low, >= rule_in high, in between indeterminate (never inverted).
    return {"balanced": float(min(youden, 1.0)), "rule_out": float(rule_out),
            "rule_in": float(max(rule_in, rule_out))}


def bootstrap_ci(fn, y, *ps, n=N_BOOT):
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], *[p[i] for p in ps]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def delong_test(y, p1, p2):
    """Paired DeLong test for two correlated AUCs (Sun & Xu fast implementation)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def components(p):
        x, yv = p[pos], p[neg]
        v10 = np.array([(np.sum(xi > yv) + 0.5 * np.sum(xi == yv)) / n for xi in x])
        v01 = np.array([(np.sum(x > yj) + 0.5 * np.sum(x == yj)) / m for yj in yv])
        return v10.mean(), v10, v01

    a1, v10a, v01a = components(p1)
    a2, v10b, v01b = components(p2)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(a1), "auc_b": float(a2), "diff": float(a1 - a2), "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}


# ---------------------------------------------------------------- efficiency
def efficiency(model_path):
    import tensorflow as tf
    from utils.model_architecture import CUSTOM_OBJECTS

    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    x = np.random.rand(1, 224, 224, 3).astype("float32")
    infer = tf.function(lambda t: model(t, training=False))
    for _ in range(5):
        infer(x)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        infer(x)
        times.append((time.perf_counter() - t0) * 1000)
    return {"params": int(model.count_params()), "flops": count_flops(model),
            "size_mb": round(Path(model_path).stat().st_size / 1e6, 2),
            "latency_ms_cpu": round(float(np.median(times)), 2)}


def count_flops(model):
    """2 x multiply-accumulates of conv / depthwise-separable / dense layers (the dominant cost)."""
    import tensorflow as tf
    from utils.model_architecture import DepthwiseSeparableConv

    flops = 0
    for layer in model.layers:
        if isinstance(layer, DepthwiseSeparableConv):
            _, h, w, cin = layer.input.shape
            k = layer.kernel_size
            flops += 2 * h * w * cin * (k * k + layer.filters)
        elif isinstance(layer, tf.keras.layers.Conv2D):
            _, h, w, cout = layer.output.shape
            kh, kw = layer.kernel_size
            flops += 2 * h * w * cout * layer.kernel.shape[2] * kh * kw
        elif isinstance(layer, tf.keras.layers.Dense):
            flops += 2 * int(np.prod(layer.kernel.shape))
    return int(flops)


# ---------------------------------------------------------------- localization
def pointing_game(model_path, batch_size=32):
    """
    Grad-CAM vs the radiologist's nodule box on the test set. `hit_rate` is the share of images whose
    heatmap peak falls inside the box; `energy_in_box` is the share of positive heatmap mass inside it,
    compared with `box_area_fraction` (what a uniform heatmap would score).
    """
    import tensorflow as tf
    from training.dataset import load_cache
    from utils.gradcam import find_target_layer
    from utils.model_architecture import CUSTOM_OBJECTS

    data = load_cache("tn5000")
    images, boxes = data["test"][0], data["test_boxes"]
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(find_target_layer(model)).output, model.output])
    hits, energy, area = [], [], []
    for start in range(0, len(images), batch_size):
        x = np.repeat(images[start:start + batch_size, ..., None].astype(np.float32) / 255.0, 3, axis=-1)
        with tf.GradientTape() as tape:
            conv, p = grad_model(x, training=False)
            p = tf.clip_by_value(p[:, 0], 1e-6, 1 - 1e-6)
            logit = tf.math.log(p / (1 - p))
            # Explain each image's predicted class; samples are independent, so one tape serves the batch.
            target = tf.reduce_sum(tf.where(p >= 0.5, logit, -logit))
        grads = tape.gradient(target, conv)
        cams = tf.nn.relu(tf.reduce_sum(conv * tf.reduce_mean(grads, axis=(1, 2), keepdims=True), axis=-1))
        cams = tf.image.resize(cams[..., None], (224, 224))[..., 0].numpy()
        for cam, (x0, y0, x1, y1) in zip(cams, boxes[start:start + batch_size]):
            mask = np.zeros_like(cam, dtype=bool)
            mask[int(max(0, y0)):int(np.ceil(y1)), int(max(0, x0)):int(np.ceil(x1))] = True
            peak = np.unravel_index(np.argmax(cam), cam.shape)
            hits.append(bool(mask[peak]))
            energy.append(float(cam[mask].sum() / cam.sum()) if cam.sum() > 0 else 0.0)
            area.append(float(mask.mean()))
    return {"n": len(hits), "hit_rate": float(np.mean(hits)), "energy_in_box": float(np.mean(energy)),
            "box_area_fraction": float(np.mean(area))}


# ---------------------------------------------------------------- recipe sweep
def _val_curve_from_log(path):
    return [float(line.split("val_auc: ")[1].split()[0]) for line in path.read_text().splitlines() if "val_auc: " in line]


def sweep_summary():
    """
    Recipe selection record. Uses VALIDATION AUROC only; sweep runs are never scored on test.
    Includes the first recipe (plateau LR, lr 1e-3), stopped at epoch 32 when it was still improving slowly.
    """
    rows = []
    v1 = LOG_ARCHIVE / "recipe_v1_fib_seed0.log"
    if v1.exists():
        va = _val_curve_from_log(v1)
        rows.append({"name": "v1_plateau1e-3_std", "model": "fibonacci_net", "seed": 0,
                     "recipe": {"lr": 1e-3, "schedule": "plateau", "aug": "standard", "epochs": 50},
                     "best_val_auc": max(va), "best_epoch": int(np.argmax(va)) + 1, "epochs_run": len(va),
                     "note": "stopped manually at epoch 32 (still improving slowly)", "val_auc_curve": va})
    for h in sorted(SWEEP_DIR.glob("sweep_*/seed*/history.json")):
        d = json.loads(h.read_text())
        va = d["history"]["val_auc"]
        rows.append({"name": h.parent.parent.name.removeprefix("sweep_"), "model": d["model"], "seed": d["seed"],
                     "recipe": d.get("recipe"), "best_val_auc": max(va), "best_epoch": int(np.argmax(va)) + 1,
                     "epochs_run": len(va), "val_auc_curve": va})
    if not rows:
        return None
    summary = {"selection_metric": "best validation AUROC (test set untouched)",
               "selected": max(rows, key=lambda r: r["best_val_auc"])["name"], "runs": rows}
    SWEEP_PATH.parent.mkdir(exist_ok=True)
    SWEEP_PATH.write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- archiving
def archive_runs(names):
    """Copies the small per-run artifacts (history, predictions) into reports/ so results ship with the repo."""
    for name in names:
        for run in sorted((RUNS_DIR / name).glob("seed*")):
            dest = ARCHIVE_DIR / name / run.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in ("history.json", "history.csv", "preds.npz"):
                if (run / f).exists():
                    shutil.copy(run / f, dest / f)
    LOG_ARCHIVE.mkdir(parents=True, exist_ok=True)
    for log in Path("logs/sweep").glob("*.log"):
        shutil.copy(log, LOG_ARCHIVE / f"sweep_{log.name}")
    for log in Path("logs/runs").glob("*.log"):
        shutil.copy(log, LOG_ARCHIVE / log.name)


def error_analysis(run_dir, calibration, threshold, top=15):
    """Most confident test errors of the deployed model, for manual review of failure modes."""
    d = np.load(run_dir / "preds.npz")
    if "files_test" not in d:
        return None
    p, y, files = apply_calibration(d["p_test"], calibration), d["y_test"], d["files_test"]
    fn = np.where((y == 1) & (p < threshold))[0]
    fp = np.where((y == 0) & (p >= threshold))[0]
    pick = lambda idx, key: [{"file": str(files[i]), "probability": round(float(p[i]), 4)}
                             for i in idx[np.argsort(key(p[idx]))][:top]]
    return {"threshold": threshold,
            "missed_malignant": pick(fn, lambda v: v),
            "false_alarms": pick(fp, lambda v: -v)}


# ---------------------------------------------------------------- main
def report_status(models):
    """'complete' only when every architecture has all seeds trained for the full recipe budget."""
    problems = []
    for name in DISPLAY:
        m = models.get(name)
        if m is None:
            problems.append(f"{name}: not trained yet")
            continue
        if m["n_seeds"] < EXPECTED_SEEDS:
            problems.append(f"{name}: {m['n_seeds']}/{EXPECTED_SEEDS} seeds")
        budget = (m.get("recipe") or {}).get("epochs")
        short = [r["seed"] for r in m["runs"] if budget and r["epochs_run"] < budget]
        if short:
            problems.append(f"{name}: seeds {short} stopped before the {budget}-epoch budget")
    return {"state": "complete" if not problems else "interim", "issues": problems}


def evaluate_run(run_dir):
    d = np.load(run_dir / "preds.npz")
    p_val, y_val, p_test, y_test = d["p_val"], d["y_val"], d["p_test"], d["y_test"]
    cal = fit_platt(p_val, y_val)
    pc_val, pc_test = apply_calibration(p_val, cal), apply_calibration(p_test, cal)
    thr = pick_thresholds(pc_val, y_val)
    e, bins = ece(pc_test, y_test)
    fpr, tpr, _ = roc_curve(y_test, pc_test)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int))
    hist = json.loads((run_dir / "history.json").read_text())
    return {
        "seed": hist["seed"], "epochs_run": hist["epochs_run"], "train_seconds": hist["train_seconds"],
        "recipe": hist.get("recipe"), "provenance": hist.get("provenance"),
        "val_auc": float(roc_auc_score(y_val, p_val)),
        "auc": float(roc_auc_score(y_test, p_test)),
        "auc_ci": bootstrap_ci(roc_auc_score, y_test, p_test),
        "auprc": float(average_precision_score(y_test, p_test)),
        "calibration": cal,
        "brier_uncalibrated": float(np.mean((p_test - y_test) ** 2)),
        "brier": float(np.mean((pc_test - y_test) ** 2)),
        "ece": e, "calibration_bins": bins,
        "thresholds": thr,
        "balanced": confusion_metrics(pc_test, y_test, thr["balanced"]),
        "rule_out": confusion_metrics(pc_test, y_test, thr["rule_out"]),
        "rule_in": confusion_metrics(pc_test, y_test, thr["rule_in"]),
        "indeterminate_fraction": float(np.mean((pc_test >= thr["rule_out"]) & (pc_test < thr["rule_in"]))),
        "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        "history": {k: hist["history"][k] for k in ("auc", "val_auc", "loss", "val_loss") if k in hist["history"]},
        "external": external_metrics(d, cal, thr) if "p_ext" in d else None,
        "_p_test": p_test, "_y_test": y_test,
        "_p_ext": d["p_ext"] if "p_ext" in d else None, "_y_ext": d["y_ext"] if "y_ext" in d else None,
    }


def external_metrics(d, calibration, thresholds):
    """External test with calibration and thresholds frozen from the development data."""
    p, y = d["p_ext"], d["y_ext"]
    pc = apply_calibration(p, calibration)
    e, bins = ece(pc, y)
    fpr, tpr, _ = roc_curve(y, pc)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int))
    return {"n": int(len(y)), "prevalence": float(y.mean()),
            "auc": float(roc_auc_score(y, p)), "auc_ci": bootstrap_ci(roc_auc_score, y, p),
            "auprc": float(average_precision_score(y, p)), "brier": float(np.mean((pc - y) ** 2)), "ece": e,
            "calibration_bins": bins,
            "balanced": confusion_metrics(pc, y, thresholds["balanced"]),
            "rule_out": confusion_metrics(pc, y, thresholds["rule_out"]),
            "rule_in": confusion_metrics(pc, y, thresholds["rule_in"]),
            "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}}


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def main():
    if "--sweep-only" in __import__("sys").argv:
        print(json.dumps({k: v for k, v in (sweep_summary() or {}).items() if k != "runs"}, indent=2))
        return
    models = {}
    for name in DISPLAY:
        runs = sorted((RUNS_DIR / name).glob("seed*/preds.npz"))
        if not runs:
            continue
        results = [evaluate_run(r.parent) for r in runs]
        # Deployment candidate per architecture: best VALIDATION AUC (never test).
        best = max(results, key=lambda r: r["val_auc"])
        best_dir = RUNS_DIR / name / f"seed{best['seed']}"
        eff = efficiency(best_dir / "model.keras")
        summary = {
            "auc": mean_sd([r["auc"] for r in results]),
            "auprc": mean_sd([r["auprc"] for r in results]),
            "brier": mean_sd([r["brier"] for r in results]),
            "ece": mean_sd([r["ece"] for r in results]),
            **{f"{op}_{k}": mean_sd([r[op][k] for r in results])
               for op in ("balanced", "rule_out", "rule_in") for k in ("sensitivity", "specificity", "ppv", "npv", "accuracy")},
            "train_minutes": mean_sd([r["train_seconds"] / 60 for r in results]),
        }
        if all(r["external"] for r in results):
            summary.update({f"external_{k}": mean_sd([r["external"][k] for r in results]) for k in ("auc", "brier", "ece")})
            summary.update({f"external_rule_out_{k}": mean_sd([r["external"]["rule_out"][k] for r in results])
                            for k in ("sensitivity", "specificity", "ppv", "npv")})
        models[name] = {"display": DISPLAY[name], "n_seeds": len(results), "summary": summary,
                        "recipe": best["recipe"],
                        "efficiency": eff, "localization": pointing_game(best_dir / "model.keras"),
                        "selected_seed": best["seed"],
                        "selected": {k: v for k, v in best.items() if not k.startswith("_")},
                        "runs": [{k: r[k] for k in ("seed", "val_auc", "auc", "epochs_run")} for r in results],
                        "_runs": results}

    comparison = None
    if PRIMARY in models and BASELINE in models:
        per_seed = []
        fib_runs = {r["seed"]: r for r in models[PRIMARY]["_runs"]}
        base_runs = {r["seed"]: r for r in models[BASELINE]["_runs"]}
        for s in sorted(set(fib_runs) & set(base_runs)):
            y = fib_runs[s]["_y_test"]
            per_seed.append({"seed": s, **delong_test(y, fib_runs[s]["_p_test"], base_runs[s]["_p_test"])})
        # Seed-averaged predictions: a variance-reduced comparison of the two architectures.
        common = sorted(set(fib_runs) & set(base_runs))
        y = fib_runs[common[0]]["_y_test"]
        pf = np.mean([fib_runs[s]["_p_test"] for s in common], axis=0)
        pb = np.mean([base_runs[s]["_p_test"] for s in common], axis=0)
        diff_ci = bootstrap_ci(lambda yy, a, b: roc_auc_score(yy, a) - roc_auc_score(yy, b), y, pf, pb)
        external = None
        if fib_runs[common[0]]["_p_ext"] is not None:
            ye = fib_runs[common[0]]["_y_ext"]
            pfe = np.mean([fib_runs[s]["_p_ext"] for s in common], axis=0)
            pbe = np.mean([base_runs[s]["_p_ext"] for s in common], axis=0)
            ext_ci = bootstrap_ci(lambda yy, a, b: roc_auc_score(yy, a) - roc_auc_score(yy, b), ye, pfe, pbe)
            external = {**delong_test(ye, pfe, pbe), "diff_ci95": ext_ci, "non_inferior": bool(ext_ci[0] > -NI_MARGIN)}
        fe, be = models[PRIMARY]["efficiency"], models[BASELINE]["efficiency"]
        comparison = {
            "seeds": common,
            "per_seed_delong": per_seed,
            "seed_averaged": {**delong_test(y, pf, pb), "diff_ci95": diff_ci},
            "external_seed_averaged": external,
            "non_inferiority_margin": NI_MARGIN,
            "non_inferior": bool(diff_ci[0] > -NI_MARGIN),
            "superior": bool(diff_ci[0] > 0),
            "param_reduction_x": round(be["params"] / fe["params"], 1),
            "flop_reduction_x": round(be["flops"] / fe["flops"], 1),
            "size_reduction_x": round(be["size_mb"] / fe["size_mb"], 1),
            "latency_speedup_x": round(be["latency_ms_cpu"] / fe["latency_ms_cpu"], 1),
        }

    status = report_status(models)
    for m in models.values():
        m.pop("_runs")

    archive_runs(list(models))
    sweep = sweep_summary()
    errors = None
    if PRIMARY in models:
        sel = models[PRIMARY]["selected"]
        errors = error_analysis(RUNS_DIR / PRIMARY / f"seed{sel['seed']}", sel["calibration"],
                                sel["thresholds"]["rule_out"])

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "status": status,
        "sweep": sweep,
        "error_analysis": errors,
        "protocol": {"dataset": "TN5000 (biopsy-confirmed), official split made group-safe",
                     "external_test": "Kaggle thyroid ultrasound set (no overlap with TN5000; never used for training,"
                                      " calibration or thresholds)",
                     "recipe_selection": "Kaggle validation split only (zero tuning on TN5000)",
                     "rule_out_target_sensitivity": RULE_OUT_SENS, "rule_in_target_specificity": RULE_IN_SPEC,
                     "bootstrap_resamples": N_BOOT},
        "data_audit": json.loads(Path("data/tn5000/audit.json").read_text()),
        "kaggle_audit": json.loads(Path("data/kaggle/audit.json").read_text()),
        "models": models,
        "comparison": comparison,
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"wrote {REPORT_PATH}")

    if PRIMARY in models:
        DEPLOY_DIR.mkdir(exist_ok=True)
        sel = models[PRIMARY]["selected"]
        shutil.copy(RUNS_DIR / PRIMARY / f"seed{sel['seed']}" / "model.keras", DEPLOY_DIR / "fibonacci_net.keras")
        (DEPLOY_DIR / "fibonacci_net.json").write_text(json.dumps({
            "seed": sel["seed"], "calibration": sel["calibration"], "thresholds": sel["thresholds"],
            "test_metrics": {op: sel[op] for op in ("balanced", "rule_out", "rule_in")}, "test_auc": sel["auc"],
            "test_auc_ci": sel["auc_ci"],
            "external_auc": sel["external"]["auc"] if sel["external"] else None,
            "external_auc_ci": sel["external"]["auc_ci"] if sel["external"] else None,
        }, indent=2))
        print(f"exported {DEPLOY_DIR / 'fibonacci_net.keras'} (seed {sel['seed']})")


if __name__ == "__main__":
    main()
