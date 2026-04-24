from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PART_NAMES = ["corner_1", "corner_2", "corner_3", "corner_4"]
COLORS = {
    "centroid": "#97d476",
    "instance": "#c779ff",
}


plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def short_name(model_dir: Path) -> str:
    name = model_dir.name.lower()
    if "centered_instance" in name:
        return "instance"
    if "centroid" in name:
        return "centroid"
    return model_dir.name


def load_metrics(model_dir: Path, split: str = "val") -> dict[str, Any]:
    path = next(model_dir.glob(f"metrics.{split}*.npz"))
    return np.load(path, allow_pickle=True)["metrics"].item()


def load_training_log(model_dir: Path) -> list[dict[str, str]]:
    with (model_dir / "training_log.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def load_model_summary(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    val = load_metrics(model_dir, "val")
    train = load_metrics(model_dir, "train")
    log = load_training_log(model_dir)

    val_losses = [float(r["val/loss"]) for r in log if r.get("val/loss") not in ("", None)]
    train_losses = [float(r["train/loss"]) for r in log if r.get("train/loss") not in ("", None)]

    return {
        "name": short_name(model_dir),
        "full_name": model_dir.name,
        "model_dir": str(model_dir),
        "val": val,
        "train": train,
        "log": log,
        "best_val_loss": min(val_losses) if val_losses else None,
        "last_val_loss": val_losses[-1] if val_losses else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "epochs_logged": len(log),
    }


def load_model_pair(centroid_dir: Path, instance_dir: Path) -> list[dict[str, Any]]:
    return [load_model_summary(centroid_dir), load_model_summary(instance_dir)]


def _series(rows: list[dict[str, str]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        epoch = row.get("epoch", "")
        value = row.get(key, "")
        if epoch not in ("", None) and value not in ("", None):
            xs.append(int(float(epoch)))
            ys.append(float(value))
    return xs, ys


def plot_learning_curves(models: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for model in models:
        name = model["name"]
        color = COLORS.get(name, None)
        train_x, train_y = _series(model["log"], "train/loss")
        val_x, val_y = _series(model["log"], "val/loss")
        if train_y:
            ax.plot(train_x, train_y, color=color, lw=2, alpha=0.95, label=f"{name} train")
        if val_y:
            ax.plot(val_x, val_y, color=color, lw=2, ls="--", alpha=0.95, label=f"{name} val")
    ax.set_title("How the models learned over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    return fig


def plot_validation_scoreboard(models: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    metric_names = ["mOKS", "mPCK", "PCK@5", "vis. precision", "vis. recall"]
    x = np.arange(len(metric_names))
    width = 0.36

    for i, model in enumerate(models):
        val = model["val"]
        scores = [
            float(val["mOKS"]["mOKS"]),
            float(val["pck_metrics"]["mPCK"]),
            float(val["pck_metrics"]["PCK@5"]),
            float(val["visibility_metrics"]["precision"]),
            float(val["visibility_metrics"]["recall"]),
        ]
        offset = (i - 0.5) * width
        ax.bar(x + offset, scores, width=width, color=COLORS.get(model["name"]), label=model["name"])

    ax.set_title("Validation snapshot")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_pixel_error(models: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    values = []
    labels = []
    colors = []
    for model in models:
        dists = np.asarray(model["val"]["distance_metrics"]["dists"], dtype=float).ravel()
        dists = dists[~np.isnan(dists)]
        values.append(dists)
        labels.append(model["name"])
        colors.append(COLORS.get(model["name"]))

    bp = ax.boxplot(values, patch_artist=True, widths=0.5, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_title("Validation pixel error")
    ax.set_ylabel("Pixels")
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_part_accuracy(models: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(PART_NAMES))
    width = 0.36
    for i, model in enumerate(models):
        vals = np.asarray(model["val"]["pck_metrics"]["mPCK_parts"], dtype=float)
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width=width, color=COLORS.get(model["name"]), label=model["name"])
    ax.set_title("Per-corner accuracy")
    ax.set_ylabel("mPCK")
    ax.set_xticks(x)
    ax.set_xticklabels(PART_NAMES)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def metric_table(models: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | mOKS | mPCK | PCK@5 | Vis. precision | Vis. recall | Avg error (px) | Best val loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for model in models:
        val = model["val"]
        vis = val["visibility_metrics"]
        lines.append(
            f"| {model['name']} | "
            f"{float(val['mOKS']['mOKS']):.3f} | "
            f"{float(val['pck_metrics']['mPCK']):.3f} | "
            f"{float(val['pck_metrics']['PCK@5']):.3f} | "
            f"{float(vis['precision']):.3f} | "
            f"{float(vis['recall']):.3f} | "
            f"{float(val['distance_metrics']['avg']):.3f} | "
            f"{model['best_val_loss']:.6f} |"
        )

    return "\n".join(lines)


def notebook_report_markdown(models: list[dict[str, Any]]) -> str:
    lines = [
        "## Model Report",
        "",
        "This report summarizes the saved validation metrics for the models loaded in this notebook.",
        "",
        "### General guide",
        "",
        "- For metrics like `mOKS`, `mPCK`, `PCK@5`, visibility precision, and visibility recall:",
        "  values **near 1** are better, and values **near 0** are worse.",
        "- For metrics like average pixel error and validation loss:",
        "  **lower is better**.",
        "- These metrics should always be interpreted together, not one at a time.",
        "",
        "### Metric meaning",
        "",
        "- `mOKS`: overall keypoint agreement. Near `1` is very good.",
        "- `mPCK`: fraction of keypoints placed close enough to the true location. Near `1` is very good.",
        "- `PCK@5`: fraction of keypoints within 5 pixels of the true point. Higher is better.",
        "- `Visibility precision`: when the model predicts a point is visible, how often that is correct.",
        "- `Visibility recall`: when a point is truly visible, how often the model finds it.",
        "- `Average pixel error`: average distance between prediction and label in pixels. Lower is better.",
        "- `Best val loss`: the lowest validation loss reached during training. Lower is better.",
        "",
        "### Metric table",
        "",
        metric_table(models),
        "",
        "### Notes",
        "",
        "- Very high validation scores on a very small validation set can still be misleading.",
        "- If inference on new videos looks poor, compare these scores with the actual rendered predictions.",
        "- A strong centroid model with a weaker centered-instance model often means detection is good but corner placement still needs improvement.",
    ]
    return "\n".join(lines)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def save_report_bundle(models: list[dict[str, Any]], outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "learning_curves": save_figure(plot_learning_curves(models), outdir / "learning_curves.png"),
        "validation_scoreboard": save_figure(plot_validation_scoreboard(models), outdir / "validation_scoreboard.png"),
        "pixel_error": save_figure(plot_pixel_error(models), outdir / "pixel_error.png"),
        "part_accuracy": save_figure(plot_part_accuracy(models), outdir / "part_accuracy.png"),
    }
    report_path = outdir / "notebook_report.md"
    report_path.write_text(notebook_report_markdown(models))
    outputs["report"] = report_path
    return outputs
