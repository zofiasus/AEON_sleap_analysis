"""
fit_homographies.py

Fit Top→Side homographies for each side camera using the stable
correspondence DataFrame produced by match_stable_correspondences.py.

Pipeline:
  stable_correspondences_*.csv
        ↓
  fit_homographies()        — RANSAC homography per camera, saves .npz + summary .csv
        ↓
  validate_homographies()   — reprojection error stats and plots
        ↓
  inspect_worst_frames()    — visual overlay of worst-error frames from video
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# --- Config -------------------------------------------------------------------

KEYPOINTS = ["corner_1", "corner_2", "corner_3", "corner_4"]
REF_CAMERA = "CameraTop"
SIDE_CAMERAS = ["CameraEast", "CameraNorth", "CameraSouth", "CameraWest", "CameraNest"]

# RANSAC defaults
RANSAC_REPROJ_THRESH_PX = 10.0
RANSAC_MAX_ITERS = 5000
RANSAC_CONFIDENCE = 0.9999


# --- Public API ---------------------------------------------------------------

def fit_homographies(
    input_path: Path,
    output_path: Path | None = None,
    min_overlap_s: float = 0.5,
    ref_camera: str = REF_CAMERA,
    side_cameras: list[str] = SIDE_CAMERAS,
    ransac_thresh_px: float = RANSAC_REPROJ_THRESH_PX,
    ransac_max_iters: int = RANSAC_MAX_ITERS,
    ransac_confidence: float = RANSAC_CONFIDENCE,
    min_correspondences: int = 4,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """
    Fit one homography per side camera mapping ref_camera -> side camera
    using RANSAC.

    Automatically loads the correspondence CSV from input_path using the
    standard filename: stable_correspondences_minOverlap{min_overlap_s}s.csv
    Saves outputs to output_path if provided, otherwise to input_path.

    Args:
        input_path: Directory containing stable_correspondences_*.csv.
        output_path: Directory to save outputs. Defaults to input_path.
        min_overlap_s: The min_overlap_s used when generating the CSV
                       (used to construct the expected filename).
        ref_camera: Reference camera (Top). Homography maps ref -> side.
        side_cameras: List of side cameras to fit.
        ransac_thresh_px: RANSAC reprojection threshold in pixels.
        ransac_max_iters: Max RANSAC iterations.
        ransac_confidence: RANSAC confidence level.
        min_correspondences: Minimum point pairs required to attempt fit.

    Returns:
        H_by_cam: Dict mapping camera name -> (3,3) homography matrix.
        summary_df: DataFrame with fit quality stats per camera.
    """
    input_path = Path(input_path)
    out_dir = Path(output_path) if output_path is not None else input_path

    # Auto-load correspondence CSV by standard name, fall back to glob
    csv_name = f"stable_correspondences_minOverlap{min_overlap_s:.1f}s.csv"
    csv_path = input_path / csv_name
    if not csv_path.exists():
        candidates = sorted(input_path.glob("stable_correspondences_minOverlap*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No stable_correspondences_*.csv found in {input_path}\n"
                f"Run match_stable_correspondences.py first."
            )
        csv_path = candidates[0]
        print(f"  [INFO] Exact file not found, using: {csv_path.name}")

    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path.name}  ({len(df)} placements)")
    out_dir.mkdir(parents=True, exist_ok=True)

    H_by_cam = {}
    inliers_by_cam = {}
    rows = []

    for cam in side_cameras:
        print(f"\n{'='*45}")
        print(f"Fitting: {ref_camera} → {cam}")

        src, dst, n_placements = _extract_point_pairs(df, ref_camera, cam)

        if src is None or src.shape[0] < min_correspondences:
            print(f"  [SKIP] Not enough correspondences: {0 if src is None else src.shape[0]}")
            continue

        print(f"  Placements used : {n_placements}")
        print(f"  Point pairs     : {src.shape[0]}  (×4 corners per placement)")

        H, inlier_mask = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh_px,
            maxIters=ransac_max_iters,
            confidence=ransac_confidence,
        )

        if H is None or inlier_mask is None:
            print(f"  [FAIL] findHomography returned None.")
            continue

        inlier_mask = inlier_mask.ravel().astype(bool)
        errs = _reprojection_errors(H, src, dst)

        n_total = src.shape[0]
        n_in = int(inlier_mask.sum())

        print(f"  Inliers         : {n_in}/{n_total} ({100*n_in/n_total:.1f}%)")
        print(f"  Median err (all): {np.median(errs):.2f} px")
        print(f"  Median err (in) : {np.median(errs[inlier_mask]):.2f} px")

        H_by_cam[cam] = H
        inliers_by_cam[cam] = inlier_mask

        rows.append({
            "camera": cam,
            "n_placements": n_placements,
            "n_correspondences": n_total,
            "n_inliers": n_in,
            "inlier_frac": n_in / n_total,
            "median_err_px_all": float(np.median(errs)),
            "median_err_px_inliers": float(np.median(errs[inlier_mask])) if n_in else np.nan,
            "mean_err_px_inliers": float(np.mean(errs[inlier_mask])) if n_in else np.nan,
            "ransac_thresh_px": ransac_thresh_px,
        })

    summary_df = pd.DataFrame(rows).sort_values("camera").reset_index(drop=True)

    # Save homographies
    H_path = output_path / f"homographies_{ref_camera}_to_sides_thresh{ransac_thresh_px:.0f}px.npz"
    np.savez_compressed(H_path, **{f"{cam}__H": H for cam, H in H_by_cam.items()})
    print(f"\nSaved homographies: {H_path}")

    # Save inlier masks alongside
    inlier_path = output_path / f"homography_inliers_{ref_camera}_to_sides_thresh{ransac_thresh_px:.0f}px.npz"
    np.savez_compressed(inlier_path, **{f"{cam}__inliers": m for cam, m in inliers_by_cam.items()})

    # Save summary
    csv_path = output_path / f"homography_summary_thresh{ransac_thresh_px:.0f}px.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary    : {csv_path}")

    return H_by_cam, summary_df


def load_homographies(processed_path: Path, ref_camera: str = REF_CAMERA, ransac_thresh_px: float = RANSAC_REPROJ_THRESH_PX) -> dict[str, np.ndarray]:
    """
    Load saved homography matrices from .npz.

    Returns:
        Dict mapping camera name -> (3,3) homography matrix.
    """
    path = processed_path / f"homographies_{ref_camera}_to_sides_thresh{ransac_thresh_px:.0f}px.npz"
    if not path.exists():
        raise FileNotFoundError(f"No homography file found: {path}")
    z = np.load(path)
    H_by_cam = {k.replace("__H", ""): z[k] for k in z.files}
    print(f"Loaded homographies for: {list(H_by_cam.keys())}")
    return H_by_cam


# --- Validation & visualisation -----------------------------------------------

def validate_homographies(
    df: pd.DataFrame,
    H_by_cam: dict[str, np.ndarray],
    ref_camera: str = REF_CAMERA,
) -> None:
    """
    Plot reprojection error histograms for all cameras side by side.
    Inliers in blue, outliers in red.
    """
    n = len(H_by_cam)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (cam, H) in zip(axes, H_by_cam.items()):
        src, dst, _ = _extract_point_pairs(df, ref_camera, cam)
        errs = _reprojection_errors(H, src, dst)

        ax.hist(errs, bins=40, color="steelblue", alpha=0.8, label="all")
        ax.set_title(f"{cam}\nmedian={np.median(errs):.1f}px")
        ax.set_xlabel("Reprojection error (px)")
        ax.set_ylabel("Count")

    plt.suptitle(f"{ref_camera} → Side reprojection errors", fontsize=11)
    plt.show()


def show_error_over_time(
    df: pd.DataFrame,
    H_by_cam: dict[str, np.ndarray],
    ref_camera: str = REF_CAMERA,
    ransac_thresh_px: float = RANSAC_REPROJ_THRESH_PX,
) -> None:
    """
    Scatter plot of mean reprojection error per placement over time,
    for each camera. Outlier placements highlighted in red.
    """
    n = len(H_by_cam)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3 * n), sharex=False, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (cam, H) in zip(axes, H_by_cam.items()):
        src, dst, _ = _extract_point_pairs(df, ref_camera, cam)
        errs = _reprojection_errors(H, src, dst)

        # reshape to (n_placements, 4) to get per-placement mean
        n_placements = src.shape[0] // 4
        errs_mat = errs.reshape(n_placements, 4)
        mean_err = errs_mat.mean(axis=1)

        # get time axis from df rows that have this camera
        cam_col = f"{cam}_corner_1_x"
        ref_col = f"{ref_camera}_corner_1_x"
        mask = df[cam_col].notna() & df[ref_col].notna()
        t = df.loc[mask, "time_start"].values

        outlier = mean_err > ransac_thresh_px
        ax.scatter(t[~outlier], mean_err[~outlier], s=12, color="steelblue", label="inlier", alpha=0.8)
        ax.scatter(t[outlier], mean_err[outlier], s=20, color="crimson", label="outlier", zorder=5)
        ax.axhline(ransac_thresh_px, color="crimson", lw=1.5, linestyle="--", label=f"thresh ({ransac_thresh_px}px)")
        ax.set_ylabel("mean err (px)")
        ax.set_title(cam)
        ax.legend(fontsize=7, frameon=False)

    axes[-1].set_xlabel("time_start (s)")
    plt.suptitle("Reprojection error over time per camera", fontsize=11)
    plt.show()


def inspect_worst_placements(
    df: pd.DataFrame,
    H_by_cam: dict[str, np.ndarray],
    video_root: Path,
    arena: str,
    experiment: str,
    session: str,
    chunk: str,
    ref_camera: str = REF_CAMERA,
    camera: str = "CameraEast",
    ransac_thresh_px: float = RANSAC_REPROJ_THRESH_PX,
    n_worst: int = 3,
) -> None:
    """
    For the given side camera, find the n_worst placements by reprojection
    error and show a visual overlay: reference frame + side frame + overlay
    with labeled corners (green=ref, red=side warped into ref coords).

    Args:
        df: Stable correspondence DataFrame.
        H_by_cam: Fitted homography matrices.
        video_root: Root directory of raw videos.
        arena/experiment/session/chunk: Path components for video files.
        camera: Side camera to inspect.
        n_worst: How many worst placements to visualise.
    """
    if camera not in H_by_cam:
        print(f"[ERROR] No homography for {camera}.")
        return

    H = H_by_cam[camera]
    H_inv = np.linalg.inv(H)

    src, dst, _ = _extract_point_pairs(df, ref_camera, camera)
    errs = _reprojection_errors(H, src, dst)
    n_placements = src.shape[0] // 4
    errs_mat = errs.reshape(n_placements, 4)
    mean_err = errs_mat.mean(axis=1)

    # Get placement rows and frame indices
    cam_col = f"{camera}_corner_1_x"
    ref_col = f"{ref_camera}_corner_1_x"
    mask = df[cam_col].notna() & df[ref_col].notna()
    placement_rows = df[mask].reset_index(drop=True)

    worst_idx = np.argsort(mean_err)[::-1][:n_worst]

    video_ref  = _build_video_path(video_root, arena, experiment, session, chunk, ref_camera)
    video_side = _build_video_path(video_root, arena, experiment, session, chunk, camera)

    for rank, pi in enumerate(worst_idx):
        row = placement_rows.iloc[pi]
        # Use midpoint of stable window as representative frame
        t_mid = (row["time_start"] + row["time_end"]) / 2
        frame_idx = int(t_mid * 50)  # assuming 50fps

        ref4  = _row_to_corners(row, ref_camera)   # (4,2)
        side4 = _row_to_corners(row, camera)        # (4,2)

        try:
            ref_bgr  = _read_frame(video_ref, frame_idx)
            side_bgr = _read_frame(video_side, frame_idx)
        except Exception as e:
            print(f"  [WARNING] Could not read frame {frame_idx}: {e}")
            continue

        ref_h, ref_w = ref_bgr.shape[:2]
        warped_side = cv2.warpPerspective(side_bgr, H_inv, (ref_w, ref_h))
        overlay = cv2.addWeighted(ref_bgr, 0.55, warped_side, 0.45, 0)

        # Side corners warped into ref coords
        side4_warped = cv2.perspectiveTransform(side4[None].astype(np.float32), H_inv)[0]

        ref_labeled  = _draw_corners(ref_bgr.copy(), ref4, color=(0, 200, 0), prefix="R:")
        side_labeled = _draw_corners(side_bgr.copy(), side4, color=(0, 100, 255), prefix="S:")
        ov_labeled   = _draw_corners(overlay.copy(), ref4, color=(0, 200, 0), prefix="R:")
        ov_labeled   = _draw_corners(ov_labeled, side4_warped, color=(0, 100, 255), prefix="S→R:")

        per_corner = {KEYPOINTS[k]: float(errs_mat[pi, k]) for k in range(4)}
        print(f"\n[Rank {rank+1}] placement {pi} | frame ~{frame_idx} | "
              f"mean_err={mean_err[pi]:.2f}px")
        print(f"  per-corner: {per_corner}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        axes[0].imshow(cv2.cvtColor(ref_labeled,  cv2.COLOR_BGR2RGB)); axes[0].set_title(f"{ref_camera} (frame {frame_idx})"); axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(side_labeled, cv2.COLOR_BGR2RGB)); axes[1].set_title(f"{camera}"); axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(ov_labeled,   cv2.COLOR_BGR2RGB)); axes[2].set_title("Overlay (green=Ref, blue=Side warped)"); axes[2].axis("off")
        plt.suptitle(f"{ref_camera}→{camera} | Rank {rank+1} worst | mean err={mean_err[pi]:.2f}px", fontsize=11)
        plt.show()


# --- Under the hood -----------------------------------------------------------

def _extract_point_pairs(
    df: pd.DataFrame,
    ref_camera: str,
    side_camera: str,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    Extract (4N, 2) source and destination point arrays from the
    correspondence DataFrame for a given camera pair.

    Only rows where BOTH cameras have valid corner coordinates are used.
    Returns (src, dst, n_placements) or (None, None, 0) if no valid rows.
    """
    # Build column lists for both cameras
    ref_cols  = [f"{ref_camera}_{kp}_{ax}"  for kp in KEYPOINTS for ax in ["x", "y"]]
    side_cols = [f"{side_camera}_{kp}_{ax}" for kp in KEYPOINTS for ax in ["x", "y"]]

    # Check columns exist
    missing = [c for c in ref_cols + side_cols if c not in df.columns]
    if missing:
        print(f"  [WARNING] Missing columns: {missing[:4]}...")
        return None, None, 0

    # Keep only rows where both cameras are present
    mask = df[ref_cols].notna().all(axis=1) & df[side_cols].notna().all(axis=1)
    sub = df[mask]

    if len(sub) == 0:
        return None, None, 0

    # Build (N, 4, 2) arrays then flatten to (4N, 2)
    src_arr = np.stack([
        sub[[f"{ref_camera}_{kp}_x", f"{ref_camera}_{kp}_y"]].values
        for kp in KEYPOINTS
    ], axis=1)  # (N, 4, 2)

    dst_arr = np.stack([
        sub[[f"{side_camera}_{kp}_x", f"{side_camera}_{kp}_y"]].values
        for kp in KEYPOINTS
    ], axis=1)  # (N, 4, 2)

    src = src_arr.reshape(-1, 2).astype(np.float64)  # (4N, 2)
    dst = dst_arr.reshape(-1, 2).astype(np.float64)  # (4N, 2)

    return src, dst, len(sub)


def _reprojection_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Euclidean reprojection error per point pair (px)."""
    ones = np.ones((src.shape[0], 1))
    X = np.concatenate([src, ones], axis=1)         # (M, 3)
    Y = (H @ X.T).T                                  # (M, 3)
    pred = Y[:, :2] / Y[:, 2:3]                      # dehomogenise → (M, 2)
    return np.linalg.norm(pred - dst, axis=1)         # (M,)


def _row_to_corners(row: pd.Series, camera: str) -> np.ndarray:
    """Extract (4, 2) corner array from a DataFrame row for a given camera."""
    return np.array([
        [row[f"{camera}_{kp}_x"], row[f"{camera}_{kp}_y"]]
        for kp in KEYPOINTS
    ], dtype=np.float64)


def _build_video_path(root: Path, arena: str, experiment: str, session: str, chunk: str, camera: str) -> Path:
    return root / arena / experiment / session / camera / f"{camera}_{chunk}.avi"


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def _draw_corners(
    img_bgr: np.ndarray,
    pts4: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    prefix: str = "",
) -> np.ndarray:
    """Draw labeled corner circles onto a BGR image."""
    for i, kp in enumerate(KEYPOINTS):
        x, y = float(pts4[i, 0]), float(pts4[i, 1])
        if not np.isfinite([x, y]).all():
            continue
        cv2.circle(img_bgr, (int(x), int(y)), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img_bgr, f"{prefix}{kp}", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img_bgr


# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.match_stable_correspondences import load_correspondences

    processed_path = Path("/Volumes/homes/live/zsus/aeon_sleap/abcEphysPilot01/processed/")

    df = load_correspondences(processed_path, min_overlap_s=0.5)

    H_by_cam, summary_df = fit_homographies(
        df=df,
        processed_path=processed_path,
        ransac_thresh_px=RANSAC_REPROJ_THRESH_PX,
    )

    print("\nHomography fit summary:")
    print(summary_df.to_string(index=False))

    validate_homographies(df, H_by_cam)
    show_error_over_time(df, H_by_cam)