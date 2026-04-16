from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio


NODE_ORDER = ["corner_1", "corner_2", "corner_3", "corner_4"]
NODE_COLORS = {
    "corner_1": "#1f77b4",
    "corner_2": "#ff7f0e",
    "corner_3": "#2ca02c",
    "corner_4": "#d62728",
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)

def ingest_and_report(slp_path: str | Path, camera_name: str = "camera") -> dict:
    """
    Load a prediction .slp, build the report, and return everything needed
    for notebook display.
    """
    slp_path = Path(slp_path)

    data = ingest_data(slp_path)
    report = generate_report(data)

    report["camera_name"] = camera_name
    report["data"] = data

    report["report_markdown"] = f"# {camera_name} Inference Report\n\n" + report["report_markdown"]

    return report



def ingest_data(slp_path: str | Path, export_dir: str | Path | None = None) -> dict:
    """Load a SLEAP prediction .slp and prepare data tables for plotting/export."""
    slp_path = Path(slp_path)
    if export_dir is None:
        export_dir = slp_path.parent / "exports"
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    labels = sio.load_file(slp_path)
    video = labels.videos[0]
    frame_rgb = _read_first_frame(video.filename)
    points_df = _make_points_df(labels)
    instance_df = _make_instance_df(labels)
    center_df = (
        points_df.groupby("frame_idx")[["x", "y"]]
        .mean()
        .reset_index()
        .sort_values("frame_idx")
    )

    return {
        "slp_path": slp_path,
        "labels": labels,
        "video": video,
        "video_path": Path(video.filename),
        "export_dir": export_dir,
        "frame_rgb": frame_rgb,
        "points_df": points_df,
        "instance_df": instance_df,
        "center_df": center_df,
        "node_order": NODE_ORDER,
    }


def generate_report(data: dict) -> dict:
    """Create the main inference plots and an easy summary markdown block."""
    figs = {
        "score_hist": plot_corner_score_histogram(data),
        "heatmap": plot_detection_heatmap(data),
        "trace": plot_corner_traces(data),
        # "center_trace": plot_center_trace(data),
    }

    report_md = _report_markdown(data)

    return {
        "figures": figs,
        "report_markdown": report_md,
    }


def export_video_small(data: dict, filename: str | Path | None = None, start: int | None = None, end: int | None = None) -> Path:
    """Export a smaller shareable labeled video."""
    if filename is None:
        filename = data["export_dir"] / "labeled_video_small.mp4"
    filename = Path(filename)
    sio.render_video(
        data["labels"],
        save_path=filename,
        start=start,
        end=end,
        preset="draft",
        scale=0.7,
        marker_size=4,
        line_width=2,
        crf=28,
        show_edges=True,
        show_progress=True,
    )
    return filename


def export_video_large(data: dict, filename: str | Path | None = None, start: int | None = None, end: int | None = None) -> Path:
    """Export a larger, higher-quality labeled video."""
    if filename is None:
        filename = data["export_dir"] / "labeled_video_large.mp4"
    filename = Path(filename)
    sio.render_video(
        data["labels"],
        save_path=filename,
        start=start,
        end=end,
        preset="final",
        scale=1.0,
        marker_size=5,
        line_width=2,
        show_edges=True,
        show_progress=True,
    )
    return filename


def plot_detection_heatmap(data: dict):
    points_df = data["points_df"]
    frame_rgb = data["frame_rgb"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    axes = axes.ravel()

    for ax, node_name in zip(axes, data["node_order"]):
        node_df = points_df[points_df["node_name"] == node_name]
        ax.imshow(frame_rgb)
        sc = ax.scatter(
            node_df["x"],
            node_df["y"],
            c=node_df["score"],
            s=8,
            alpha=0.35,
            cmap="inferno",
        )
        ax.set_title(node_name)
        ax.axis("off")

    fig.suptitle("Where each corner was detected", fontsize=13)
    cbar = fig.colorbar(sc, ax=axes, location="right", pad=0.02)
    cbar.set_label("Prediction score")
    return fig


def plot_corner_traces(data: dict):
    points_df = data["points_df"]
    frame_rgb = data["frame_rgb"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    axes = axes.ravel()

    for ax, node_name in zip(axes, data["node_order"]):
        node_df = points_df[points_df["node_name"] == node_name].sort_values("frame_idx")
        ax.imshow(frame_rgb)
        ax.plot(node_df["x"], node_df["y"], linewidth=1.2, alpha=0.8, color="cyan")
        ax.scatter(node_df["x"].iloc[0], node_df["y"].iloc[0], color="lime", s=30)
        ax.scatter(node_df["x"].iloc[-1], node_df["y"].iloc[-1], color="red", s=30)
        ax.set_title(f"{node_name} trace")
        ax.axis("off")

    fig.suptitle("How each corner moved through the video", fontsize=13)
    return fig


# def plot_center_trace(data: dict):
#     frame_rgb = data["frame_rgb"]
#     center_df = data["center_df"]

#     fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
#     ax.imshow(frame_rgb)
#     ax.plot(center_df["x"], center_df["y"], color="deepskyblue", linewidth=2, alpha=0.85)
#     ax.scatter(center_df["x"].iloc[0], center_df["y"].iloc[0], color="lime", s=40, label="start")
#     ax.scatter(center_df["x"].iloc[-1], center_df["y"].iloc[-1], color="red", s=40, label="end")
#     ax.set_title("Card center trace")
#     ax.legend(frameon=False)
#     ax.axis("off")
#     return fig


def plot_corner_score_histogram(data: dict):
    points_df = data["points_df"]

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    for node_name in data["node_order"]:
        node_df = points_df[points_df["node_name"] == node_name]
        ax.hist(
            node_df["score"],
            bins=30,
            alpha=0.4,
            label=node_name,
            color=NODE_COLORS[node_name],
        )
    ax.set_title("Confidence score distribution by corner")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    return fig


def _make_points_df(labels) -> pd.DataFrame:
    rows = []
    for lf in labels.labeled_frames:
        for inst_idx, inst in enumerate(lf.instances):
            for pt in inst.points:
                rows.append(
                    {
                        "frame_idx": lf.frame_idx,
                        "instance_idx": inst_idx,
                        "node_name": pt["name"],
                        "x": float(pt["xy"][0]),
                        "y": float(pt["xy"][1]),
                        "score": float(pt["score"]),
                        "visible": bool(pt["visible"]),
                    }
                )
    return pd.DataFrame(rows)


def _make_instance_df(labels) -> pd.DataFrame:
    rows = []
    for lf in labels.labeled_frames:
        for inst_idx, inst in enumerate(lf.instances):
            point_scores = [pt["score"] for pt in inst.points]
            rows.append(
                {
                    "frame_idx": lf.frame_idx,
                    "instance_idx": inst_idx,
                    "mean_score": float(np.mean(point_scores)),
                    "min_score": float(np.min(point_scores)),
                    "max_score": float(np.max(point_scores)),
                    "n_points": len(inst.points),
                }
            )
    return pd.DataFrame(rows)


def _read_first_frame(video_path: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from video: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _report_markdown(data: dict) -> str:
    instance_df = data["instance_df"]
    points_df = data["points_df"]
    frames_with_predictions = points_df["frame_idx"].nunique()
    mean_score = instance_df["mean_score"].mean()
    low_score_fraction = float((instance_df["mean_score"] < 0.2).mean()) if len(instance_df) else 0.0

    lines = [
        "## Inference Report",
        "",
        f"- Prediction file: `{data['slp_path']}`",
        f"- Video file: `{data['video_path']}`",
        f"- Frames with predictions: **{frames_with_predictions}**",
        f"- Predicted instances: **{len(instance_df)}**",
        f"- Mean instance score: **{mean_score:.3f}**",
        f"- Fraction of low-confidence instances (`mean score < 0.2`): **{low_score_fraction:.3f}**",
        "",
        "### How to read these plots",
        "",
        "- **Heatmap**: shows where each corner was detected on top of the first video frame.",
        "- **Corner traces**: shows how each corner moved through the video over time.",
        "- **Card center trace**: shows the overall path of the card through the scene.",
        "- **Confidence histogram**: shows whether the model was consistently confident or often uncertain for each corner.",
        "",
        "### Saved figures",
        "",
    ]
    return "\n".join(lines)
