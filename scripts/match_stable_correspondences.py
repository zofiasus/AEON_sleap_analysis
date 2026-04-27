from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from itertools import combinations


# --- Config -------------------------------------------------------------------

FPS = 50.0
CAMERAS = ["CameraEast", "CameraNorth", "CameraSouth", "CameraWest", "CameraNest", "CameraTop"]


# --- Public API ---------------------------------------------------------------

def match_stable_correspondences(
    cameras: list[str],
    processed_path: Path,
    output_path: Path | None = None,
    min_overlap_s: float = 0.5,
) -> pd.DataFrame:
    """
    Match stable card events across all camera pairs and build a flat
    correspondence DataFrame — one row per unique card placement, columns
    for each camera's corner coordinates (NaN where camera didn't see it).

    Args:
        cameras: List of camera names.
        processed_path: Directory containing {camera}_stable_events.nc files.
        output_path: Where to save output .csv. Defaults to processed_path.
        min_overlap_s: Minimum temporal overlap in seconds for two stable
                       events to count as the same card placement.

    Returns:
        DataFrame with one row per card placement instance.
    """
    out_dir = output_path if output_path is not None else processed_path

    # Load all stable event datasets
    ds_by_cam = _load_all_stable_events(cameras, processed_path)
    if not ds_by_cam:
        raise RuntimeError("No stable event files found — run extract_stable_events first.")

    active_cameras = list(ds_by_cam.keys())
    print(f"\nLoaded stable events for: {active_cameras}")
    for cam, ds in ds_by_cam.items():
        print(f"  {cam}: {len(ds.event)} events")

    # Find all pairwise overlapping event matches
    all_matches = _find_all_pairwise_matches(ds_by_cam, active_cameras, min_overlap_s)

    # Cluster pairwise matches into global placement instances
    placements = _cluster_into_placements(all_matches, active_cameras)
    print(f"\nUnique card placements found: {len(placements)}")

    # Build flat DataFrame
    df = _build_correspondence_df(placements, ds_by_cam, active_cameras)

    # Report coverage
    _print_coverage_summary(df, active_cameras)

    # Save
    out_file = out_dir / f"stable_correspondences_minOverlap{min_overlap_s:.1f}s.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")

    return df


def load_correspondences(processed_path: Path, min_overlap_s: float = 0.5) -> pd.DataFrame:
    """
    Load a previously saved correspondence DataFrame.

    Args:
        processed_path: Directory containing the saved .csv file.
        min_overlap_s: The min_overlap_s used when generating the file.

    Returns:
        DataFrame of stable correspondences.
    """
    path = processed_path / f"stable_correspondences_minOverlap{min_overlap_s:.1f}s.csv"
    if not path.exists():
        raise FileNotFoundError(f"No correspondence file found at {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} placements from {path.name}")
    return df


# --- Under the hood -----------------------------------------------------------

def _load_all_stable_events(
    cameras: list[str],
    processed_path: Path,
) -> dict[str, xr.Dataset]:
    """Load stable event datasets for all cameras that have a file."""
    ds_by_cam = {}
    for cam in cameras:
        nc_path = processed_path / f"{cam}_stable_events.nc"
        if not nc_path.exists():
            print(f"  [WARNING] No stable events file for {cam}, skipping.")
            continue
        ds_by_cam[cam] = xr.open_dataset(nc_path)
    return ds_by_cam


def _find_all_pairwise_matches(
    ds_by_cam: dict[str, xr.Dataset],
    cameras: list[str],
    min_overlap_s: float,
) -> dict[tuple[str, str], list[tuple[int, int, float]]]:
    """
    For every camera pair, find pairs of events that overlap in time
    by at least min_overlap_s seconds.

    Returns:
        Dict mapping (cam_a, cam_b) -> list of (event_idx_a, event_idx_b, overlap_s).
    """
    matches = {}

    for cam_a, cam_b in combinations(cameras, 2):
        if cam_a not in ds_by_cam or cam_b not in ds_by_cam:
            continue

        ds_a = ds_by_cam[cam_a]
        ds_b = ds_by_cam[cam_b]

        starts_a = ds_a["time_start"].values
        ends_a   = ds_a["time_end"].values
        starts_b = ds_b["time_start"].values
        ends_b   = ds_b["time_end"].values

        pair_matches = []
        for i in range(len(starts_a)):
            for j in range(len(starts_b)):
                overlap = min(ends_a[i], ends_b[j]) - max(starts_a[i], starts_b[j])
                if overlap >= min_overlap_s:
                    pair_matches.append((i, j, float(overlap)))

        matches[(cam_a, cam_b)] = pair_matches
        print(f"  {cam_a} ↔ {cam_b}: {len(pair_matches)} overlapping event pairs")

    return matches


def _cluster_into_placements(
    all_matches: dict[tuple[str, str], list[tuple[int, int, float]]],
    cameras: list[str],
) -> list[dict[str, int]]:
    """
    Cluster pairwise event matches into global placement instances using
    union-find. Each placement is a dict mapping camera -> event_idx for
    all cameras that observed that card placement.

    A placement groups events that are mutually overlapping across cameras,
    so one row can have coordinates from 3+ cameras simultaneously.

    Returns:
        List of dicts, each mapping camera_name -> event_index.
    """
    # Build nodes as (camera, event_idx) tuples
    # Union-find: merge nodes connected by pairwise matches
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (cam_a, cam_b), pairs in all_matches.items():
        for i, j, _ in pairs:
            union((cam_a, i), (cam_b, j))

    # Group nodes by their root
    groups = {}
    for (cam_a, cam_b), pairs in all_matches.items():
        for i, j, _ in pairs:
            for node in [(cam_a, i), (cam_b, j)]:
                root = find(node)
                if root not in groups:
                    groups[root] = {}
                cam, idx = node
                # If camera already in group, keep the one with lower idx (arbitrary tie-break)
                if cam not in groups[root]:
                    groups[root][cam] = idx

    return list(groups.values())


def _build_correspondence_df(
    placements: list[dict[str, int]],
    ds_by_cam: dict[str, xr.Dataset],
    cameras: list[str],
) -> pd.DataFrame:
    """
    Build a flat DataFrame with one row per placement.

    Columns:
        placement_id, n_cameras_seen, time_start, time_end,
        {camera}_{keypoint}_{x|y} for all cameras and keypoints,
        {camera}_mean_confidence, {camera}_duration_s, {camera}_n_frames
    """
    # Get keypoint names from first available dataset
    first_ds = next(iter(ds_by_cam.values()))
    keypoints = first_ds["keypoint"].values

    rows = []

    for placement_id, placement in enumerate(placements):
        row = {"placement_id": placement_id}

        # Temporal bounds: union of all overlapping event windows
        all_starts, all_ends = [], []
        for cam, ev_idx in placement.items():
            ds = ds_by_cam[cam]
            all_starts.append(float(ds["time_start"].values[ev_idx]))
            all_ends.append(float(ds["time_end"].values[ev_idx]))

        row["time_start"] = min(all_starts)
        row["time_end"]   = max(all_ends)
        row["n_cameras_seen"] = len(placement)

        # Per-camera coordinates and metadata
        for cam in cameras:
            if cam not in placement:
                # Camera didn't see this placement — fill with NaN
                for kp in keypoints:
                    row[f"{cam}_{kp}_x"] = np.nan
                    row[f"{cam}_{kp}_y"] = np.nan
                row[f"{cam}_mean_confidence"] = np.nan
                row[f"{cam}_duration_s"]      = np.nan
                row[f"{cam}_n_frames"]        = np.nan
                continue

            ev_idx = placement[cam]
            ds = ds_by_cam[cam]

            # position: (event, space, keypoint)
            pos = ds["position"].values[ev_idx]  # (2, 4): space x keypoint
            conf = ds["mean_confidence"].values[ev_idx]  # (4,)

            for k, kp in enumerate(keypoints):
                row[f"{cam}_{kp}_x"] = float(pos[0, k])  # space=0 is x
                row[f"{cam}_{kp}_y"] = float(pos[1, k])  # space=1 is y

            row[f"{cam}_mean_confidence"] = float(conf.mean())
            row[f"{cam}_duration_s"]      = float(ds["duration"].values[ev_idx])
            row[f"{cam}_n_frames"]        = int(ds["n_frames"].values[ev_idx])

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("time_start").reset_index(drop=True)
    return df


def _print_coverage_summary(df: pd.DataFrame, cameras: list[str]) -> None:
    """Print how many placements each camera contributed to."""
    print("\nCoverage summary:")
    print(f"  Total placements : {len(df)}")
    print(f"  Cameras seen per placement: mean={df['n_cameras_seen'].mean():.1f}  "
          f"min={df['n_cameras_seen'].min()}  max={df['n_cameras_seen'].max()}")
    print()
    for cam in cameras:
        col = f"{cam}_mean_confidence"
        if col not in df.columns:
            continue
        seen = df[col].notna().sum()
        print(f"  {cam:20s} seen in {seen}/{len(df)} placements ({100*seen/len(df):.0f}%)")


# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    processed_path = Path("/Volumes/homes/live/zsus/aeon_sleap/abcEphysPilot01/processed/")

    df = match_stable_correspondences(
        cameras=CAMERAS,
        processed_path=processed_path,
        min_overlap_s=0.5,
    )

    print("\nFirst few rows:")
    print(df.head())

    print("\nColumn names:")
    print(list(df.columns))

    # Example: filter to placements seen by at least 2 cameras
    multi = df[df["n_cameras_seen"] >= 2]
    print(f"\nPlacements seen by 2+ cameras: {len(multi)}")

    # Example: filter to placements seen by CameraTop AND CameraEast
    top_east = df[df["CameraTop_corner_1_x"].notna() & df["CameraEast_corner_1_x"].notna()]
    print(f"Placements seen by CameraTop AND CameraEast: {len(top_east)}")