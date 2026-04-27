from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# --- Config -------------------------------------------------------------------

FPS = 50.0
CAMERAS = ["CameraEast", "CameraNorth", "CameraSouth", "CameraWest", "CameraNest", "CameraTop"]


# --- Public Functions ---------------------------------------------------------

def extract_stable_events(
    cameras: list[str],
    processed_path: Path,
    output_path: Path | None = None,
    min_duration_s: float = 1.0,
    pixel_tolerance: float | None = None,
    min_area_fraction: float = 0.5,
    min_confidence: float | None = None,
    fps: float = FPS,
    visual: bool = False,
) -> Path:
    """
    Full pipeline: load pose data, find stable card events, save as .nc.
 
    For each camera, loads {camera}_pose_data.nc, extracts stable windows
    where all 4 corners are detected and not moving, and saves the result
    as {camera}_stable_events.nc.
 
    Args:
        cameras: List of camera names.
        processed_path: Directory containing {camera}_pose_data.nc files.
        output_path: Where to save output. Defaults to processed_path.
        min_duration_s: Minimum stable window duration in seconds.
        pixel_tolerance: Max per-corner displacement to count as stable (pixels).
                         If None, inferred from displacement distribution.
        min_area_fraction: Card quadrilateral area must be > this fraction of
                           the median valid area to pass geometry check.
        min_confidence: Minimum confidence per keypoint. If None, inferred
                        as Q1 of valid confidence values for that camera.
        fps: Frame rate of the data.
        visual: If True, plot displacement and confidence for each camera.
 
    Returns:
        Path to the output directory.
    """
    out_dir = output_path if output_path is not None else processed_path
    out_dir.mkdir(parents=True, exist_ok=True)
 
    for cam in cameras:
        print(f"\n{'='*50}")
        print(f"Processing: {cam}")
        print(f"{'='*50}")
 
        nc_path = processed_path / f"{cam}_pose_data.nc"
        if not nc_path.exists():
            print(f"  [WARNING] File not found, skipping: {nc_path}")
            continue
 
        ds = xr.open_dataset(nc_path)
 
        # Infer thresholds if not provided
        conf_threshold = (
            min_confidence
            if min_confidence is not None
            else _infer_confidence_threshold(ds, cam)
        )
 
        # Run pipeline
        valid_mask = _filter_valid_frames(ds, conf_threshold)
        print(f"  Valid frames (all corners + confidence): {valid_mask.sum()}/{len(valid_mask)}")
 
        geom_mask = _check_card_geometry(ds, valid_mask, min_area_fraction)
        print(f"  Geometry-passing frames: {geom_mask.sum()}/{valid_mask.sum()}")
 
        displacement = _compute_displacement(ds, geom_mask)
 
        tol = (
            pixel_tolerance
            if pixel_tolerance is not None
            else _infer_pixel_tolerance(displacement)
        )
 
        static_mask = displacement < tol
        windows = _find_stable_windows(static_mask, geom_mask, min_duration_s, fps)
        print(f"  Stable windows found: {len(windows)}")
 
        if visual:
            _show_stable_events(ds, displacement, windows, tol, conf_threshold, cam)
 
        if len(windows) == 0:
            print(f"  [WARNING] No stable events found for {cam}, skipping save.")
            continue
 
        events_ds = _build_events_dataset(
            ds=ds,
            windows=windows,
            camera=cam,
            fps=fps,
            pixel_tolerance=tol,
            min_duration_s=min_duration_s,
            min_confidence=conf_threshold,
        )
 
        out_file = out_dir / f"{cam}_stable_events.nc"
        events_ds.to_netcdf(out_file)
        print(f"  Saved: {out_file}")
 
    return out_dir


def load_stable_events(
    cameras: list[str],
    processed_path: Path,
) -> dict[str, xr.Dataset]:
    """
    Load saved stable event .nc files for each camera.

    Args:
        cameras: List of camera names.
        processed_path: Directory containing {camera}_stable_events.nc files.

    Returns:
        Dict mapping camera name -> xr.Dataset.
    """
    ds_by_cam = {}

    for cam in cameras:
        nc_path = processed_path / f"{cam}_stable_events.nc"
        if not nc_path.exists():
            print(f"  [WARNING] File not found, skipping: {nc_path}")
            continue
        ds_by_cam[cam] = xr.open_dataset(nc_path)
        print(f"Loaded {cam}: {nc_path.name}  ({len(ds_by_cam[cam].event)} events)")

    return ds_by_cam


# --- Under the hood -----------------------------------------------------------

def _infer_confidence_threshold(ds: xr.Dataset, camera: str) -> float:
    vals = ds["confidence"].values.flatten()
    valid = vals[~np.isnan(vals)]

    if len(valid) == 0:
        print(f"  [WARNING] No valid confidence values, threshold set to 0.0")
        return 0.0

    # Use Q1 to exclude the bottom 25% of detections
    threshold = float(np.percentile(valid, 25))
    threshold = min(threshold, 1.0)  # cap at 100%

    print(f"  Confidence distribution (valid n={len(valid)}):")
    print(f"    Q1={np.percentile(valid,25):.4f}  Q2={np.percentile(valid,50):.4f}  Q3={np.percentile(valid,75):.4f}  max={valid.max():.4f}")
    print(f"  Inferred confidence threshold (Q1): {threshold:.4f}")
    return threshold


def _filter_valid_frames(ds: xr.Dataset, min_confidence: float) -> np.ndarray:
    """
    Return boolean mask of frames where all 4 corners are detected
    and all confidence values exceed the threshold.

    Returns:
        Boolean array of shape (time,).
    """
    conf = ds["confidence"].values  # (time, keypoints, individuals)

    # All keypoints must be non-NaN
    all_detected = ~np.any(np.isnan(conf), axis=(1, 2))

    # All keypoints must meet confidence threshold
    above_threshold = np.all(conf >= min_confidence, axis=(1, 2))

    return all_detected & above_threshold


def _check_card_geometry(
    ds: xr.Dataset,
    valid_mask: np.ndarray,
    min_area_fraction: float,
) -> np.ndarray:
    """
    Check that detected corners form a plausible card quadrilateral.
    Filters on area: must be > min_area_fraction * median valid area.

    Returns:
        Boolean array of shape (time,).
    """
    pos = ds["position"].values  # (time, space, keypoints, individuals)
    # shape: (time, 2, 4, 1) → squeeze individuals → (time, 2, 4)
    pos = pos[..., 0]

    areas = np.full(len(valid_mask), np.nan)

    valid_indices = np.where(valid_mask)[0]
    for t in valid_indices:
        pts = pos[t].T  # (4, 2): 4 corners, x and y
        areas[t] = _quad_area(pts)

    # Median area from valid frames as reference
    valid_areas = areas[valid_indices]
    median_area = np.nanmedian(valid_areas)
    print(f"  Median card area (valid frames): {median_area:.1f} px²")

    area_threshold = min_area_fraction * median_area
    geom_mask = (areas >= area_threshold) & valid_mask

    return geom_mask


def _quad_area(pts: np.ndarray) -> float:
    """
    Compute area of a quadrilateral using the shoelace formula.
    pts: (4, 2) array of corner coordinates.
    """
    x, y = pts[:, 0], pts[:, 1]
    n = len(x)
    area = 0.5 * abs(
        sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n))
    )
    return area


def _compute_displacement(ds: xr.Dataset, geom_mask: np.ndarray) -> np.ndarray:
    """
    Compute mean frame-to-frame displacement across all 4 corners,
    only on geometry-passing frames.

    Returns:
        Array of shape (time,) with displacement in pixels.
        Non-geometry frames set to np.inf (will not pass static threshold).
    """
    pos = ds["position"].values  # (time, space, keypoints, individuals)
    pos = pos[..., 0]            # (time, 2, 4)

    n_frames = pos.shape[0]
    displacement = np.full(n_frames, np.inf)

    valid_indices = np.where(geom_mask)[0]

    for i in range(1, len(valid_indices)):
        t_prev = valid_indices[i - 1]
        t_curr = valid_indices[i]

        # Only compare consecutive frames (skip gaps)
        if t_curr - t_prev > 1:
            continue

        diff = pos[t_curr] - pos[t_prev]          # (2, 4)
        per_corner_dist = np.linalg.norm(diff, axis=0)  # (4,)
        displacement[t_curr] = per_corner_dist.mean()

    return displacement


def _infer_pixel_tolerance(displacement: np.ndarray) -> float:
    """
    Infer pixel tolerance from the displacement distribution.
    Uses the 10th percentile of finite displacements as the stable threshold.
    """
    finite_disp = displacement[np.isfinite(displacement)]

    if len(finite_disp) == 0:
        print("  [WARNING] No finite displacements found, defaulting tolerance to 2.0 px")
        return 2.0

    tolerance = float(np.median(finite_disp) + 2 * np.std(finite_disp))
    print(f"  Inferred pixel tolerance (P10 of displacements): {tolerance:.3f} px")
    return tolerance


def _find_stable_windows(
    static_mask: np.ndarray,
    geom_mask: np.ndarray,
    min_duration_s: float,
    fps: float,
) -> list[tuple[int, int]]:
    """
    Find runs of consecutive static frames exceeding min_duration_s.

    Args:
        static_mask: Boolean array, True where displacement < tolerance.
        geom_mask: Boolean array, True where geometry is valid.
        min_duration_s: Minimum window duration in seconds.
        fps: Frame rate.

    Returns:
        List of (start_frame, end_frame) tuples (inclusive).
    """
    min_frames = int(min_duration_s * fps)
    combined = static_mask & geom_mask

    windows = []
    in_window = False
    start = 0

    for t, val in enumerate(combined):
        if val and not in_window:
            start = t
            in_window = True
        elif not val and in_window:
            duration = t - start
            if duration >= min_frames:
                windows.append((start, t - 1))
            in_window = False

    # Catch window that runs to end of array
    if in_window:
        duration = len(combined) - start
        if duration >= min_frames:
            windows.append((start, len(combined) - 1))

    return windows


def _extract_event_coords(
    ds: xr.Dataset,
    windows: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract median corner coordinates and mean confidence per stable window.

    Returns:
        positions: (n_events, 2, 4) median x/y per corner
        confidences: (n_events, 4) mean confidence per corner
    """
    pos = ds["position"].values[..., 0]   # (time, 2, 4)
    conf = ds["confidence"].values[..., 0] # (time, 4)

    positions = []
    confidences = []

    for start, end in windows:
        positions.append(np.median(pos[start:end + 1], axis=0))     # (2, 4)
        confidences.append(np.mean(conf[start:end + 1], axis=0))    # (4,)

    return np.stack(positions), np.stack(confidences)


def _build_events_dataset(
    ds: xr.Dataset,
    windows: list[tuple[int, int]],
    camera: str,
    fps: float,
    pixel_tolerance: float,
    min_duration_s: float,
    min_confidence: float,
) -> xr.Dataset:
    """
    Assemble stable events into an xarray Dataset.
    """
    time_coord = ds["time"].values
    keypoints = ds["keypoints"].values
    space = ds["space"].values

    positions, confidences = _extract_event_coords(ds, windows)

    n_events = len(windows)
    time_start = np.array([time_coord[s] for s, _ in windows])
    time_end = np.array([time_coord[e] for _, e in windows])
    duration = time_end - time_start
    n_frames = np.array([e - s + 1 for s, e in windows])

    events_ds = xr.Dataset(
        {
            "position": (["event", "space", "keypoint"], positions),
            "mean_confidence": (["event", "keypoint"], confidences),
            "time_start": (["event"], time_start),
            "time_end": (["event"], time_end),
            "duration": (["event"], duration),
            "n_frames": (["event"], n_frames),
        },
        coords={
            "event": np.arange(n_events),
            "space": space,
            "keypoint": keypoints,
        },
        attrs={
            "camera": camera,
            "fps": fps,
            "pixel_tolerance": pixel_tolerance,
            "min_duration_s": min_duration_s,
            "min_confidence": min_confidence,
            "source_file": str(ds.attrs.get("source_file", "unknown")),
        },
    )

    return events_ds

def _show_stable_events(
    ds: xr.Dataset,
    displacement: np.ndarray,
    windows: list[tuple[int, int]],
    pixel_tolerance: float,
    conf_threshold: float,
    camera: str,
) -> None:
    """
    Plot displacement and per-keypoint confidence over time, with stable
    windows highlighted in green. Called by extract_stable_events when
    visual=True.
 
    Args:
        ds: Raw pose xarray Dataset for this camera.
        displacement: Frame-to-frame displacement array (time,).
        windows: List of (start_frame, end_frame) stable window tuples.
        pixel_tolerance: Displacement threshold used (shown as red dashed line).
        conf_threshold: Confidence threshold used (shown as red dashed line).
        camera: Camera name for plot title.
    """
    time = ds["time"].values
    conf = ds["confidence"].values[..., 0]  # (time, keypoints)
    keypoints = ds["keypoints"].values
 
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
 
    # --- Top: displacement ---
    disp_plot = np.where(np.isfinite(displacement), displacement, np.nan)
    axes[0].plot(time, disp_plot, lw=0.5, color="steelblue", label="displacement")
    axes[0].axhline(pixel_tolerance, color="red", lw=1.5, linestyle="--",
                    label=f"tolerance ({pixel_tolerance:.2f}px)")
    axes[0].set_ylabel("mean displacement (px)")
    axes[0].legend(fontsize=8)
 
    # --- Bottom: confidence per keypoint ---
    for i, kp in enumerate(keypoints):
        axes[1].plot(time, conf[:, i], lw=0.5, label=kp)
    axes[1].axhline(conf_threshold, color="red", lw=1.5, linestyle="--",
                    label=f"Q1 ({conf_threshold:.2f})")
    axes[1].set_ylabel("confidence")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=7)
 
    # --- Shade stable windows on both axes ---
    for s, e in windows:
        for ax in axes:
            ax.axvspan(time[s], time[e], alpha=0.2, color="green")
 
    plt.suptitle(f"{camera} — stable windows (green)")
    plt.tight_layout()
    plt.show()
 


# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    processed_path = Path("/Volumes/homes/live/zsus/aeon_sleap/abcEphysPilot01/processed/")

    out_dir = extract_stable_events(
        cameras=CAMERAS,
        processed_path=processed_path,
        min_duration_s=1.0,
        pixel_tolerance=None,     # infer from data
        min_confidence=None,      # infer as Q3
        min_area_fraction=0.5,
        fps=FPS,
    )

    ds_by_cam = load_stable_events(
        cameras=CAMERAS,
        processed_path=out_dir,
    )

    for cam, ds in ds_by_cam.items():
        print(f"\n{cam}: {len(ds.event)} stable events")
        print(f"  Durations: min={ds.duration.values.min():.2f}s  max={ds.duration.values.max():.2f}s  mean={ds.duration.values.mean():.2f}s")