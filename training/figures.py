"""
Renders the README benchmark figure from reports/benchmark.json.

    python -m training.figures
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path("reports/benchmark.json")
OUT = Path("reports/figures/benchmark.png")
COLORS = {"fibonacci_net": "#2a78d6", "baseline_cnn": "#eb6834"}
NAMES = {"fibonacci_net": "FibonacciNet", "baseline_cnn": "Standard CNN"}


def main():
    r = json.loads(REPORT.read_text())
    m = r["models"]
    keys = ["fibonacci_net", "baseline_cnn"]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (acc, cost) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})

    # Accuracy: AUROC mean ± SD over seeds on both test sets.
    groups = [("auc", "TN5000 test"), ("external_auc", "External test")]
    x = np.arange(len(groups))
    for i, k in enumerate(keys):
        means = [m[k]["summary"][g]["mean"] for g, _ in groups]
        sds = [m[k]["summary"][g]["sd"] for g, _ in groups]
        bars = acc.bar(x + (i - 0.5) * 0.36, means, 0.34, yerr=sds, capsize=3, color=COLORS[k], label=NAMES[k])
        acc.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=8, fontsize=9)
    acc.set_xticks(x, [label for _, label in groups])
    acc.set_ylim(0.5, 1.0)
    acc.set_ylabel("AUROC (0.5 = chance; mean ± SD, 3 seeds)")
    acc.set_title("Accuracy: equivalent", loc="left", fontweight="bold")
    acc.legend(frameon=False, loc="upper right")

    # Cost: each metric relative to the standard CNN (= 100%).
    metrics = [("params", "Parameters", lambda v: f"{v / 1e6:.2f} M"),
               ("flops", "FLOPs / image", lambda v: f"{v / 1e9:.2f} G"),
               ("latency_ms_cpu", "CPU time / image", lambda v: f"{v:.1f} ms")]
    y = np.arange(len(metrics))
    for i, k in enumerate(keys):
        vals = [m[k]["efficiency"][key] for key, _, _ in metrics]
        rel = [v / m["baseline_cnn"]["efficiency"][key] * 100 for v, (key, _, _) in zip(vals, metrics)]
        bars = cost.barh(y + (i - 0.5) * 0.36, rel, 0.34, color=COLORS[k])
        cost.bar_label(bars, labels=[fmt(v) for v, (_, _, fmt) in zip(vals, metrics)], padding=4, fontsize=9)
    cost.set_yticks(y, [label for _, label, _ in metrics])
    cost.invert_yaxis()
    cost.set_xlim(0, 125)
    cost.set_xlabel("% of standard CNN")
    cost.set_title("Cost: a fraction of the standard CNN", loc="left", fontweight="bold")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")

    clinical(r, keys)


def clinical(r, keys):
    """ROC curves on both test sets and FibonacciNet's calibration (validation-selected runs)."""
    m = r["models"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for ax, ext, title in ((axes[0], False, "ROC, TN5000 test"), (axes[1], True, "ROC, external test")):
        for k in keys:
            run = m[k]["selected"]["external"] if ext else m[k]["selected"]
            ax.plot(run["roc"]["fpr"], run["roc"]["tpr"], color=COLORS[k], lw=2, label=NAMES[k])
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
        ax.set_xlabel("1 − specificity (benign flagged)")
        ax.set_ylabel("Sensitivity (cancers caught)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower right")

    sel = m["fibonacci_net"]["selected"]
    ax = axes[2]
    for bins, label, color in ((sel["calibration_bins"], "TN5000 test", COLORS["fibonacci_net"]),
                               (sel["external"]["calibration_bins"], "External test", "#8a8a8a")):
        ax.plot([b["pred"] for b in bins], [b["obs"] for b in bins], "o-", color=color, label=label)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_xlabel("Predicted probability of malignancy")
    ax.set_ylabel("Observed fraction malignant")
    ax.set_title("FibonacciNet calibration", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    out = OUT.parent / "clinical.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
