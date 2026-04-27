from movement.io import load_dataset
from pathlib import Path
import xarray as xr
import numpy as np
 
 
# --- Config -------------------------------------------------------------------
 
REMOTE_PATH = Path("/Volumes/homes/live/zsus/aeon_sleap/abcEphysPilot01/")
PRED_PATH = Path(REMOTE_PATH, "predictions/")
FPS = 50.0

CAMERAS = ["CameraEast", "CameraNorth", "CameraSouth", "CameraWest", "CameraNest", "CameraTop"]
 
 
# --- Ingest -------------------------------------------------------------------
 
def ingest_pose_data(
    cameras: list[str],
    pred_path: Path,
    fps: float = FPS,
    timestamp: str | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Full ingestion pipeline: find .slp files, load into xarray, save as .nc.
 
    If timestamp is provided, matches exact file: {camera}_{camera}_{timestamp}.slp
    Otherwise globs all .slp files per camera and loads the first match.
    Output defaults to pred_path.parent/processed/ if not specified.
 
    Args:
        cameras: List of camera names.
        pred_path: Directory containing .slp prediction files.
        fps: Frame rate to assign to datasets. Defaults to module-level FPS.
        timestamp: Optional timestamp string for exact file matching.
        output_path: Directory to save .nc files. Defaults to pred_path.parent/processed/.
 
    Returns:
        Path to the output directory.
    """
    out_dir = output_path if output_path is not None else pred_path.parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
 
    for cam in cameras:
        slp_path = _resolve_slp_path(pred_path, cam, timestamp)
        if slp_path is None:
            continue
 
        print(f"Ingesting {cam}: {slp_path.name}")
        ds = load_dataset(slp_path, source_software="auto", fps=fps)

        out_file = out_dir / f"{cam}_pose_data.nc"
        
        if out_file.exists():
            print(f"  Skipping {cam}: already ingested")
            continue

        ds.to_netcdf(out_file)

        print(f"  Saved: {out_file}")
 
    return out_dir
 
 
def _resolve_slp_path(pred_path: Path, camera: str, timestamp: str | None) -> Path | None:
    """
    Resolve the .slp file path for a given camera.
    Private helper — not intended to be called directly.
    """
    if timestamp:
        path = pred_path / f"{camera}_{camera}_{timestamp}.slp"
        if not path.exists():
            print(f"  [WARNING] File not found, skipping: {path}")
            return None
        return path
 
    matches = sorted(pred_path.glob(f"{camera}*.slp"))
    if not matches:
        print(f"  [WARNING] No .slp files found for {camera} in {pred_path}")
        return None
    return matches[0]
 
 
# --- Load ---------------------------------------------------------------------
 
def load_pose_datasets(
    cameras: list[str],
    processed_path: Path,
) -> dict[str, xr.Dataset]:
    """
    Load saved .nc files for each camera into a dict of xarray Datasets.
 
    Args:
        cameras: List of camera names.
        processed_path: Directory containing {camera}_label_data.nc files.
 
    Returns:
        Dict mapping camera name -> xr.Dataset.
    """
    ds_by_cam = {}
 
    for cam in cameras:
        nc_path = processed_path / f"{cam}_pose_data.nc"
        if not nc_path.exists():
            print(f"  [WARNING] File not found, skipping: {nc_path}")
            continue
        ds_by_cam[cam] = xr.open_dataset(nc_path)
        print(f"Loaded {cam}: {nc_path.name}")
 
    return ds_by_cam
 
 
# --- Describe -----------------------------------------------------------------
 
def describe_datasets(ds_by_cam: dict[str, xr.Dataset]) -> None:
    """
    Print confidence summary stats for all cameras in the dataset dict.
 
    Args:
        ds_by_cam: Dict mapping camera name -> xr.Dataset.
    """
    for cam in ds_by_cam:
        describe_camera(ds_by_cam, cam)
 
 
def describe_camera(ds_by_cam: dict[str, xr.Dataset], camera: str) -> None:
    """
    Print summary stats for a single camera dataset.

    Args:
        ds_by_cam: Dict mapping camera name -> xr.Dataset.
        camera: Camera name to describe.
    """
    if camera not in ds_by_cam:
        print(f"[ERROR] Camera '{camera}' not found. Available: {list(ds_by_cam.keys())}")
        return

    ds = ds_by_cam[camera]
    vals = ds["confidence"].values.flatten()
    total = len(vals)
    n_nan = np.sum(np.isnan(vals))
    n_valid = total - n_nan
    time = ds.time.values

    # Individuals: frames where at least one keypoint has a valid confidence
    conf = ds["confidence"]  # (time, keypoints, individuals)
    individuals = ds.coords["individuals"].values
    keypoints = ds.coords["keypoints"].values

    print(f"\n{camera}:")
    print(f"  Source      : {ds.attrs.get('source_file', 'unknown')}")
    print(f"  FPS         : {ds.attrs.get('fps', 'unknown')}")
    print(f"  Frames      : {len(time)}  ({time[0]:.2f}s — {time[-1]:.2f}s)")
    print(f"  Individuals : {len(individuals)}  {list(individuals)}")
    print(f"  Keypoints   : {len(keypoints)}  {list(keypoints)}")
    print(f"  NaN         : {n_nan}/{total} ({100*n_nan/total:.1f}%)")
    print(f"  Valid       : {n_valid}/{total} ({100*n_valid/total:.1f}%)")

    # Per-individual detection rate (frames where any keypoint is valid)
    for ind in individuals:
        ind_conf = conf.sel(individuals=ind).values  # (time, keypoints)
        detected_frames = np.any(~np.isnan(ind_conf), axis=1).sum()
        print(f"  {ind} detected in {detected_frames}/{len(time)} frames ({100*detected_frames/len(time):.1f}%)")

    # Per-keypoint detection rate
    print(f"  Keypoint detection rates:")
    for kp in keypoints:
        kp_conf = conf.sel(keypoints=kp).values.flatten()
        kp_valid = np.sum(~np.isnan(kp_conf))
        print(f"    {kp:20s} {kp_valid}/{len(time)} ({100*kp_valid/len(time):.1f}%)")

    if n_valid > 0:
        v = vals[~np.isnan(vals)]
        print(f"  Confidence  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")
 
# --- Main ---------------------------------------------------------------------
 
if __name__ == "__main__":
    # Step 1: ingest .slp files and save as .nc (run once)
    out_dir = ingest_pose_data(
        cameras=CAMERAS,
        pred_path=PRED_PATH,
        fps=FPS,
        timestamp=None,    # set to None to glob all .slp files
    )
 
    # Step 2: load saved .nc files
    ds_by_cam = load_pose_datasets(
        cameras=CAMERAS,
        processed_path=out_dir,
    )
 
    # Step 3: inspect
    describe_datasets(ds_by_cam)               # all cameras
    describe_camera(ds_by_cam, "CameraEast")   # single camera