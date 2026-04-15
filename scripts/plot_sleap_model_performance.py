from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PART_NAMES = ["corner_1", "corner_2", "corner_3", "corner_4"]


def load_metrics(model_dir: Path, split: str) -> dict[str, Any]:
    return np.load(model_dir / f"metrics.{split}.0.npz", allow_pickle=True)["metrics"].item()


def load_log(model_dir: Path) -> list[dict[str, str]]:
    with (model_dir / "training_log.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def nonempty_series(rows: list[dict[str, str]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        value = row.get(key, "")
        epoch = row.get("epoch", "")
        if value not in ("", None) and epoch not in ("", None):
            xs.append(int(float(epoch)))
            ys.append(float(value))
    return xs, ys


def summarize(model_dir: Path) -> dict[str, Any]:
    val = load_metrics(model_dir, "val")
    train = load_metrics(model_dir, "train")
    log = load_log(model_dir)
    return {
        "name": model_dir.name,
        "dir": str(model_dir),
        "val": val,
        "train": train,
        "log": log,
    }


def short_name(model_name: str) -> str:
    if "centroid" in model_name:
        return "centroid"
    if "centered_instance" in model_name:
        return "instance"
    return model_name


def plot_training_curves(models: list[dict[str, Any]], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        epochs_t, train_loss = nonempty_series(model["log"], "train/loss")
        epochs_v, val_loss = nonempty_series(model["log"], "val/loss")
        label = short_name(model["name"])
        if train_loss:
            ax.plot(epochs_t, train_loss, marker="o", linewidth=2, label=f"{label} train")
        if val_loss:
            ax.plot(epochs_v, val_loss, marker="s", linewidth=2, linestyle="--", label=f"{label} val")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = outdir / "training_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_distance_distribution(models: list[dict[str, Any]], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for model in models:
        dists = np.asarray(model["val"]["distance_metrics"]["dists"], dtype=float).ravel()
        dists = dists[~np.isnan(dists)]
        data.append(dists)
        labels.append(short_name(model["name"]))
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for body, color in zip(parts["bodies"], ["#1f77b4", "#ff7f0e"]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pixel error")
    ax.set_title("Validation Pixel Error Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = outdir / "distance_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pck_curves(models: list[dict[str, Any]], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        thresholds = np.asarray(model["val"]["pck_metrics"]["thresholds"], dtype=float)
        pcks = np.asarray(model["val"]["pck_metrics"]["pcks"], dtype=float)
        curve = pcks.mean(axis=(0, 1))
        ax.plot(thresholds, curve, marker="o", linewidth=2, label=short_name(model["name"]))
    ax.set_title("How Fast Accuracy Improves as Error Tolerance Increases")
    ax.set_xlabel("Distance threshold (pixels)")
    ax.set_ylabel("Fraction of keypoints correct")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = outdir / "pck_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_part_accuracy(models: list[dict[str, Any]], outdir: Path) -> Path:
    x = np.arange(len(PART_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, model in enumerate(models):
        vals = np.asarray(model["val"]["pck_metrics"]["mPCK_parts"], dtype=float)
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=short_name(model["name"]))
    ax.set_xticks(x)
    ax.set_xticklabels(PART_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-part mPCK")
    ax.set_title("Which Corners Are Easy vs Fragile")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = outdir / "part_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scoreboard(models: list[dict[str, Any]], outdir: Path) -> Path:
    metric_names = ["mOKS", "mPCK", "PCK@5", "visibility precision", "visibility recall"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metric_names))
    width = 0.35
    for idx, model in enumerate(models):
        val = model["val"]
        scores = [
            float(val["mOKS"]["mOKS"]),
            float(val["pck_metrics"]["mPCK"]),
            float(val["pck_metrics"]["PCK@5"]),
            float(val["visibility_metrics"]["precision"]),
            float(val["visibility_metrics"]["recall"]),
        ]
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(x + offset, scores, width=width, label=short_name(model["name"]))
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Quick Validation Scoreboard")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = outdir / "scoreboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_summary(models: list[dict[str, Any]], outdir: Path, figure_paths: list[Path]) -> Path:
    summary = []
    for model in models:
        val = model["val"]
        log = model["log"]
        val_losses = [float(r["val/loss"]) for r in log if r.get("val/loss") not in ("", None)]
        summary.append({
            "model": model["name"],
            "val_mOKS": float(val["mOKS"]["mOKS"]),
            "val_mPCK": float(val["pck_metrics"]["mPCK"]),
            "val_PCK_at_5": float(val["pck_metrics"]["PCK@5"]),
            "avg_pixel_error": float(val["distance_metrics"]["avg"]),
            "per_part_mPCK": np.asarray(val["pck_metrics"]["mPCK_parts"], dtype=float).tolist(),
            "visibility_precision": float(val["visibility_metrics"]["precision"]),
            "visibility_recall": float(val["visibility_metrics"]["recall"]),
            "best_val_loss": min(val_losses) if val_losses else None,
            "frames_in_val_metrics": len(val["distance_metrics"]["frame_idxs"]),
            "val_frame_idxs": list(val["distance_metrics"]["frame_idxs"]),
        })
    md = ["# First Model Performance Pass", "", "These plots compare the centroid and centered-instance parts of the current best-performing SLEAP pipeline.", ""]
    md.append("## Read This First")
    md.append("")
    md.append("- The centroid model is the detector: it finds where the card is.")
    md.append("- The centered-instance model is the localizer: it refines the card corners once the centroid has been found.")
    md.append("- The saved SLEAP validation metrics here are based on only 2 validation frames, so the plots are great for diagnosis but not the final word.")
    md.append("")
    md.append("## Figures")
    md.append("")
    for path in figure_paths:
        md.append(f"### {path.stem.replace('_', ' ').title()}")
        md.append(f"![{path.stem}]({path})")
        md.append("")
    md.append("## Metric Summary")
    md.append("")
    md.append("```json")
    md.append(json.dumps(summary, indent=2))
    md.append("```")
    out = outdir / "first_model_performance_report.md"
    out.write_text("\n".join(md))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dirs", nargs="+", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    models = [summarize(path.resolve()) for path in args.model_dirs]
    figures = [
        plot_scoreboard(models, args.outdir),
        plot_training_curves(models, args.outdir),
        plot_distance_distribution(models, args.outdir),
        plot_pck_curves(models, args.outdir),
        plot_part_accuracy(models, args.outdir),
    ]
    report = write_summary(models, args.outdir, figures)
    print(report)
    for fig in figures:
        print(fig)


if __name__ == "__main__":
    main()
