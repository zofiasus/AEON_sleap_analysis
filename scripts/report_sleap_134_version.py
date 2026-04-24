from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PART_NAMES = ["corner_1", "corner_2", "corner_3", "corner_4"]
COLORS = {"centroid": "#2e86abb8", "instance": "#e09c3da9"}


def short_name(model_name: str) -> str:
    if "centroid" in model_name and "centered" not in model_name:
        return "centroid"
    if "centered_instance" in model_name:
        return "instance"
    return model_name


def _load_metrics(model_dir: Path, split: str) -> Dict[str, Any]:
    return np.load(model_dir / f"metrics.{split}.npz", allow_pickle=True)["metrics"].item()


def _load_log(model_dir: Path) -> List[Dict[str, str]]:
    with (model_dir / "training_log.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def _series(rows: List[Dict[str, str]], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for row in rows:
        value = row.get(key, "")
        epoch = row.get("epoch", "")
        if value not in ("", None) and epoch not in ("", None):
            xs.append(int(float(epoch)))
            ys.append(float(value))
    return xs, ys


def load_model_summary(model_dir: Path) -> Dict[str, Any]:
    val = _load_metrics(model_dir, "val")
    train = _load_metrics(model_dir, "train")
    log = _load_log(model_dir)
    return {
        "name": short_name(model_dir.name),
        "dir": str(model_dir),
        "val": val,
        "train": train,
        "log": log,
    }


def load_model_pair(centroid_dir: Path, instance_dir: Path) -> List[Dict[str, Any]]:
    return [load_model_summary(centroid_dir), load_model_summary(instance_dir)]


def plot_learning_curves(models: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for model in models:
        name = model["name"]
        color = COLORS.get(name)
        train_x, train_y = _series(model["log"], "loss")
        val_x, val_y = _series(model["log"], "val_loss")
        if train_y:
            ax.plot(train_x, train_y, color=color, lw=2, alpha=0.95, label=f"{name} train")
        if val_y:
            ax.plot(val_x, val_y, color=color, lw=2, ls="--", alpha=0.95, label=f"{name} val")
    ax.set_title("How the models learned over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_validation_scoreboard(models: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    metric_names = ["mOKS", "mPCK", "PCK@5", "vis. precision", "vis. recall"]
    x = np.arange(len(metric_names))
    width = 0.36
    for i, model in enumerate(models):
        val = model["val"]
        pcks = np.asarray(val["pck.pcks"], dtype=float)
        pck_at_5 = float(pcks[:, :, 4].mean())
        scores = [
            float(val["oks.mOKS"]),
            float(val["pck.mPCK"]),
            pck_at_5,
            float(val["vis.precision"]),
            float(val["vis.recall"]),
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


def plot_pixel_error(models: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    values = []
    labels = []
    colors = []
    for model in models:
        dists = np.asarray(model["val"]["dist.dists"], dtype=float).ravel()
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


def plot_part_accuracy(models: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(PART_NAMES))
    width = 0.36
    for i, model in enumerate(models):
        vals = np.asarray(model["val"]["pck.mPCK_parts"], dtype=float)
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


def metric_table(models: List[Dict[str, Any]]) -> str:
    header = "| Metric | " + " | ".join(m["name"] for m in models) + " |"
    sep = "|---|" + "---|" * len(models)
    rows = []
    for metric, getter in [
        ("mOKS",             lambda v: f"{float(v['oks.mOKS']):.3f}"),
        ("mPCK",             lambda v: f"{float(v['pck.mPCK']):.3f}"),
        ("PCK@5",            lambda v: f"{float(np.asarray(v['pck.pcks'])[:,:,4].mean()):.3f}"),
        ("vis. precision",   lambda v: f"{float(v['vis.precision']):.3f}"),
        ("vis. recall",      lambda v: f"{float(v['vis.recall']):.3f}"),
        ("avg pixel error",  lambda v: f"{float(v['dist.avg']):.2f}px"),
    ]:
        vals = " | ".join(getter(m["val"]) for m in models)
        rows.append(f"| {metric} | {vals} |")
    return "\n".join([header, sep] + rows)

def notebook_report_markdown(models: List[Dict[str, Any]]) -> str:
    lines = [
        "## Model Report",
        "",
        "### Metric guide",
        "",
        "- `mOKS`, `mPCK`, `PCK@5`, visibility precision/recall: **near 1 is better**",
        "- Average pixel error, val loss: **lower is better**",
        "",
        "### Metric table",
        "",
        metric_table(models),
        "",
        "### Notes",
        "",
        "- Centroid model = detector. Centered-instance model = corner localiser.",
        "- High scores on a small val set can be misleading — always check rendered predictions too.",
    ]
    return "\n".join(lines)

def plot_all(models: List[Dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # --- learning curves ---
    ax = axes[0, 0]
    for model in models:
        name = model["name"]
        color = COLORS.get(name)
        train_x, train_y = _series(model["log"], "loss")
        val_x, val_y = _series(model["log"], "val_loss")
        if train_y:
            ax.plot(train_x, train_y, color=color, lw=2, alpha=0.95, label=f"{name} train")
        if val_y:
            ax.plot(val_x, val_y, color=color, lw=2, ls="--", alpha=0.95, label=f"{name} val")
    ax.set_title("Learning curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)

    # --- scoreboard ---
    ax = axes[0, 1]
    metric_names = ["mOKS", "mPCK", "PCK@5", "vis. prec", "vis. recall"]
    x = np.arange(len(metric_names))
    width = 0.36
    for i, model in enumerate(models):
        val = model["val"]
        pcks = np.asarray(val["pck.pcks"], dtype=float)
        pck_at_5 = float(pcks[:, :, 4].mean())
        scores = [
            float(val["oks.mOKS"]),
            float(val["pck.mPCK"]),
            pck_at_5,
            float(val["vis.precision"]),
            float(val["vis.recall"]),
        ]
        offset = (i - 0.5) * width
        ax.bar(x + offset, scores, width=width, color=COLORS.get(model["name"]), label=model["name"])
    ax.set_title("Validation scoreboard")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    # --- pixel error ---
    ax = axes[1, 0]
    values, labels, colors = [], [], []
    for model in models:
        dists = np.asarray(model["val"]["dist.dists"], dtype=float).ravel()
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

    # --- part accuracy ---
    ax = axes[1, 1]
    x = np.arange(len(PART_NAMES))
    for i, model in enumerate(models):
        vals = np.asarray(model["val"]["pck.mPCK_parts"], dtype=float)
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width=width, color=COLORS.get(model["name"]), label=model["name"])
    ax.set_title("Per-corner accuracy")
    ax.set_ylabel("mPCK")
    ax.set_xticks(x)
    ax.set_xticklabels(PART_NAMES)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle("SLEAP Model Report — 260424.abcEphysPilot01", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig