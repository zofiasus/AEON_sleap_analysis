"""
follow_the_mouse.py

Generates a composite video that:
  - Switches between side cameras (N/S/E/W) based on where the tracked
    object is in the Top view, using a precomputed preference map
  - Crops and follows the object in the selected camera view
  - Shows picture-in-picture (PiP) insets from Nest/Patch cameras when
    the object enters defined ROIs

Inputs (from previous pipeline steps):
  - Homography .npz  (from fit_homographies.py)
  - SLEAP .slp file  (for object position in Top view)
  - ROI definitions  (defined manually in notebook, saved as JSON)

All coordinate systems are in Top camera pixel space unless noted.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# --- Config -------------------------------------------------------------------

FPS_OUT        = 30
GRID_STEP_PX   = 10        # preference map resolution
CROP_FRACTION  = 0.25      # crop window = this fraction of side camera dims
INTERP_WINDOW  = 15        # frames over which to interpolate missing positions
PIP_SCALE      = 0.28      # PiP inset as fraction of output frame width
PIP_MARGIN_PX  = 12        # gap from frame edge
PIP_BORDER_PX  = 3         # coloured border thickness around inset
BASE_FPS       = 50.0      # reference fps (Top + side cameras)

# Rotation to apply per camera so all views share East camera orientation
# West = reference (no rotation)
# North = 90° clockwise, South = 90° anti-clockwise, East = 180°
CAMERA_ROTATION = {
    "CameraEast":  cv2.ROTATE_180,                     
    "CameraNorth": cv2.ROTATE_90_CLOCKWISE,
    "CameraSouth": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "CameraWest":  None,                            # reference orientation
}
 

NODE_ORDER = ["corner_1", "corner_2", "corner_3", "corner_4"]
NODE_TO_I  = {n: i for i, n in enumerate(NODE_ORDER)}


# --- Data classes -------------------------------------------------------------

@dataclass
class PlexiLine:
    """
    A straight line in Top pixel coords dividing two sets of cameras.
    Defined by two points p1 and p2.
    Cameras listed in 'left_cameras' are only eligible for points on the
    left side of the directed line p1→p2, and vice versa.
    Pass None to disable plexiglass constraint.
    """
    p1: tuple[float, float]
    p2: tuple[float, float]
    left_cameras:  list[str] = field(default_factory=list)
    right_cameras: list[str] = field(default_factory=list)


@dataclass
class ROI:
    """
    A region of interest in Top pixel coords.
    When the object centroid enters this region, a PiP inset from
    `camera` is shown.
    """
    name:   str
    camera: str
    color:  tuple[int, int, int] = (255, 255, 255)   # BGR for PiP border

    # Rect ROI — defined by centre + dimensions
    center_x:  float = None
    center_y:  float = None
    width_px:  float = None
    height_px: float = None

    # Circle ROI
    cx: float  = None
    cy: float  = None
    radius: float = None

    # Derived rect corners (computed from center+dims)
    @property
    def x0(self): return self.center_x - self.width_px  / 2 if self.center_x is not None else None
    @property
    def y0(self): return self.center_y - self.height_px / 2 if self.center_y is not None else None
    @property
    def x1(self): return self.center_x + self.width_px  / 2 if self.center_x is not None else None
    @property
    def y1(self): return self.center_y + self.height_px / 2 if self.center_y is not None else None

    def contains(self, xy: tuple[float, float]) -> bool:
        x, y = float(xy[0]), float(xy[1])
        if self.center_x is not None:  # rect
            return (self.x0 <= x <= self.x1) and (self.y0 <= y <= self.y1)
        if self.cx is not None:        # circle
            return np.hypot(x - self.cx, y - self.cy) <= self.radius
        return False

    @staticmethod
    def rect(name, camera, center_x, center_y, width_px, height_px, color=(255, 255, 255)):
        """Define a rectangular ROI by its centre point and dimensions."""
        return ROI(name=name, camera=camera, color=color,
                   center_x=center_x, center_y=center_y,
                   width_px=width_px, height_px=height_px)

    @staticmethod
    def circle(name, camera, cx, cy, radius, color=(255, 255, 255)):
        """Define a circular ROI by its centre and radius."""
        return ROI(name=name, camera=camera, color=color, cx=cx, cy=cy, radius=radius)


# --- Public API ---------------------------------------------------------------

def get_video_info(video_path: Path) -> dict:
    """
    Read fps, frame count, and dimensions directly from a video file header.
 
    Args:
        video_path: Path to .avi (or any OpenCV-readable video).
 
    Returns:
        Dict with keys: fps, frame_count, width, height, duration_s.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open: {video_path}")
 
    info = {
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_s":  cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return info
 
 
def get_camera_fps(
    cameras: list[str],
    video_root: Path,
    arena: str,
    experiment: str,
    session: str,
    chunk: str,
    base_fps: float = BASE_FPS,
) -> dict[str, float]:
    """
    Read fps for each camera directly from its .avi file header.
    Cameras whose file is not found fall back to base_fps.
 
    Also updates the module-level CAMERA_FPS dict so _cam_frame_idx
    works without needing to pass fps_map explicitly everywhere.
 
    Args:
        cameras: List of all camera names to check.
        video_root/arena/experiment/session/chunk: Video path components.
        base_fps: Fallback fps for cameras whose file is not found.
 
    Returns:
        Dict mapping camera name -> fps.
    """
    global CAMERA_FPS
    fps_map = {}
 
    print("Reading camera fps from file headers:")
    for cam in cameras:
        path = _build_video_path(video_root, arena, experiment, session, chunk, cam)
        if not path.exists():
            fps_map[cam] = base_fps
            print(f"  {cam:20s}  [not found] → fallback {base_fps:.1f} fps")
            continue
        try:
            info = get_video_info(path)
            fps_map[cam] = info["fps"]
            print(f"  {cam:20s}  fps={info['fps']:.1f}  "
                  f"{info['width']}x{info['height']}  "
                  f"T={info['frame_count']}  ({info['duration_s']:.1f}s)")
        except Exception as e:
            fps_map[cam] = base_fps
            print(f"  {cam:20s}  [error: {e}] → fallback {base_fps:.1f} fps")
 
    CAMERA_FPS = fps_map
    return fps_map

def build_preference_map(
    homography_path: Path,
    top_frame: np.ndarray,
    side_cameras: list[str],
    plexi_line: PlexiLine | None = None,
    grid_step_px: int = GRID_STEP_PX,
    cam_wh: dict[str, tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a grid-resolution map over the Top frame assigning each grid cell
    to the best side camera (closest to centre of that camera's view).
 
    Optionally respects a plexiglass dividing line: cameras on the wrong
    side of the line are excluded from consideration for each grid point.
 
    Args:
        homography_path: Path to homographies .npz (from fit_homographies.py).
        top_frame: One BGR frame from the Top camera (used for dimensions).
        side_cameras: Ordered list of side camera names.
        plexi_line: Optional PlexiLine constraint. Pass None to ignore.
        grid_step_px: Resolution of the preference grid in pixels.
        cam_wh: Dict of camera -> (width, height). If None, loaded from npz attrs
                (you must pass it explicitly if not stored there).
 
    Returns:
        best_cam_idx: (n_gy, n_gx) int array, index into side_cameras. -1 = none.
        xs: 1D array of grid x positions.
        ys: 1D array of grid y positions.
    """
    top_h, top_w = top_frame.shape[:2]
    z = np.load(homography_path)
 
    H_top_to_cam = {}
    for cam in side_cameras:
        key = f"{cam}__H"
        if key not in z:
            print(f"  [WARNING] Key {key} not found in {homography_path.name}, skipping {cam}")
            continue
        H_top_to_cam[cam] = z[key]
 
    xs = np.arange(0, top_w, grid_step_px, dtype=int)
    ys = np.arange(0, top_h, grid_step_px, dtype=int)
    best_cam_idx = np.full((len(ys), len(xs)), -1, dtype=int)
    best_score   = np.full((len(ys), len(xs)), np.inf, dtype=float)
 
    for yi, y in enumerate(ys):
        pts_top = np.stack(
            [xs.astype(float), np.full_like(xs, float(y))], axis=1
        )  # (n_gx, 2)
 
        for ci, cam in enumerate(side_cameras):
            if cam not in H_top_to_cam or cam not in (cam_wh or {}):
                continue
 
            # Plexiglass constraint: skip this camera if the point is on the
            # wrong side of the dividing line
            eligible = np.ones(len(xs), dtype=bool)
            if plexi_line is not None:
                side = _side_of_line(pts_top, plexi_line.p1, plexi_line.p2)
                if cam in plexi_line.left_cameras:
                    eligible = side <= 0
                elif cam in plexi_line.right_cameras:
                    eligible = side >= 0
 
            if not eligible.any():
                continue
 
            pts_cam = _apply_homography(H_top_to_cam[cam], pts_top)  # (n_gx, 2)
            w, h = cam_wh[cam]
 
            for xi in range(len(xs)):
                if not eligible[xi]:
                    continue
                p = pts_cam[xi]
                # if not _in_bounds(p, w, h, margin=5.0): # lack makes it fill in gaps
                #     continue
                score = _center_distance_norm(p, w, h)
                if score < best_score[yi, xi]:
                    best_score[yi, xi] = score
                    best_cam_idx[yi, xi] = ci
 
    print(f"Preference map built: {len(ys)}×{len(xs)} grid over {top_w}×{top_h} px")
    return best_cam_idx, xs, ys
 
 
def visualise_preference_map(
    best_cam_idx: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    top_frame: np.ndarray,
    side_cameras: list[str],
    rois: list[ROI] | None = None,
    plexi_line: PlexiLine | None = None,
) -> None:
    """
    Show the preference map overlaid on the Top camera frame.
    Each camera gets a distinct colour; ROI boundaries drawn on top.
    """
    top_h, top_w = top_frame.shape[:2]
    top_rgb = cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB)
 
    palette = [
        np.array([255, 80,  80]),   # red-ish
        np.array([80,  200, 80]),   # green-ish
        np.array([80,  80,  255]),  # blue-ish
        np.array([255, 220, 50]),   # yellow-ish
        np.array([200, 80,  255]),  # purple-ish
    ]
 
    color_map = np.zeros((len(ys), len(xs), 3), dtype=np.uint8)
    for ci in range(len(side_cameras)):
        mask = best_cam_idx == ci
        color_map[mask] = palette[ci % len(palette)]
 
    assign_full = cv2.resize(color_map, (top_w, top_h), interpolation=cv2.INTER_NEAREST)
    overlay = (0.6 * top_rgb + 0.4 * assign_full).astype(np.uint8)
 
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title("Camera preference map (overlaid on Top view)")
 
    # Legend
    handles = [
        mpatches.Patch(color=palette[i % len(palette)] / 255, label=cam)
        for i, cam in enumerate(side_cameras)
    ]
 
    # Draw ROIs
    if rois:
        for roi in rois:
            if roi.center_x is not None:
                rect = mpatches.Rectangle(
                    (roi.x0, roi.y0), roi.width_px, roi.height_px,
                    linewidth=2, edgecolor="white", facecolor="none", linestyle="--"
                )
                ax.add_patch(rect)
                ax.text(roi.x0, roi.y0 - 8, f"{roi.name}", color="white", fontsize=9)
            elif roi.cx is not None:
                circ = mpatches.Circle(
                    (roi.cx, roi.cy), roi.radius,
                    linewidth=2, edgecolor="white", facecolor="none", linestyle="--"
                )
                ax.add_patch(circ)
                ax.text(roi.cx + roi.radius + 4, roi.cy,
                        f"{roi.name}", color="white", fontsize=9)
        handles.append(mpatches.Patch(color="white", label="ROIs"))
 
    # Draw plexiglass line
    if plexi_line is not None:
        x1, y1 = plexi_line.p1
        x2, y2 = plexi_line.p2
        ax.plot([x1, x2], [y1, y2], color="cyan", linewidth=2.5,
                linestyle="-", label="Plexiglass")
        handles.append(mpatches.Patch(color="cyan", label="Plexiglass boundary"))
 
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()
 
 
def select_camera_per_frame(
    centers_top: np.ndarray,
    centers_valid: np.ndarray,
    best_cam_idx: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    side_cameras: list[str],
    rois: list[ROI] | None = None,
    grid_step_px: int = GRID_STEP_PX,
) -> np.ndarray:
    """
    For each frame, select the best camera based on:
      1. If object centroid is inside a hard ROI → that ROI's camera
         (ROIs only affect PiP in this pipeline, not main camera selection,
          but you can add override logic here if needed)
      2. Otherwise → preference map lookup
      3. If no valid detection → carry forward last known camera
 
    Returns:
        selected_cam: (T,) array of camera name strings.
    """
    T = len(centers_top)
    selected_cam = np.full(T, "", dtype=object)
    last_cam = side_cameras[0]
 
    for t in range(T):
        if not centers_valid[t]:
            selected_cam[t] = last_cam
            continue
 
        xy = centers_top[t]
        cam = _lookup_best_cam(xy, best_cam_idx, xs, ys, side_cameras, grid_step_px)
        selected_cam[t] = cam if cam is not None else last_cam
        last_cam = selected_cam[t]
 
    return selected_cam
 
 
def load_object_positions(
    nc_path: Path,
    T: int,
    conf_threshold: float = 0.6,
    interp_window: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load object centroid from a processed {camera}_pose_data.nc file
    instead of reading the raw .slp directly.
    """
    ds = xr.open_dataset(nc_path)
    
    pos  = ds["position"].values   # (time, space, keypoints, individuals)
    conf = ds["confidence"].values # (time, keypoints, individuals)
    
    # Valid = all 4 corners present + all above threshold
    all_finite  = np.isfinite(pos).all(axis=(1, 2, 3))
    min_conf    = np.nanmin(conf[..., 0], axis=1)   # (time,)
    raw_valid   = all_finite & (min_conf >= conf_threshold)
    
    # Centroid = mean of 4 corners in x and y
    centers = np.full((T, 2), np.nan, dtype=np.float32)
    pos_valid = pos[raw_valid, :, :, 0]             # (n_valid, space, keypoints)
    centers[raw_valid, 0] = pos_valid[:, 0, :].mean(axis=1)  # x
    centers[raw_valid, 1] = pos_valid[:, 1, :].mean(axis=1)  # y
    
    centers, valid = _interpolate_centers(centers, raw_valid, interp_window)
    print(f"  NC: {int(raw_valid.sum())} raw valid → {int(valid.sum())} after interpolation (T={T})")
    return centers, valid
 
 
def render_composite_video(
    output_path: Path,
    selected_cam: np.ndarray,
    centers_top: np.ndarray,
    centers_valid: np.ndarray,
    homography_path: Path,
    video_root: Path,
    arena: str,
    experiment: str,
    session: str,
    chunk: str,
    side_cameras: list[str],
    rois: list[ROI] | None = None,
    t_start: int = 0,
    t_end: int | None = None,
    fps_out: float = FPS_OUT,
    crop_fraction: float = CROP_FRACTION,
    pip_scale: float = PIP_SCALE,
    pip_margin_px: int = PIP_MARGIN_PX,
    pip_border_px: int = PIP_BORDER_PX,
    annotate: bool = True,
    camera_fps: dict[str, float] | None = None,
    base_fps: float = BASE_FPS,
    centre_dot: bool = False,
) -> None:
    """
    Render the composite video:
      - Main frame: selected side camera, cropped around the projected object position
      - PiP insets: any ROI cameras active at that frame (object inside ROI)
 
    Args:
        output_path: Path to write .mp4.
        selected_cam: (T,) camera name per frame.
        centers_top: (T,2) object centroid in Top coords.
        centers_valid: (T,) validity mask.
        homography_path: Path to homographies .npz.
        video_root/arena/experiment/session/chunk: Video path components.
        side_cameras: List of side camera names.
        rois: List of ROI objects for PiP triggers.
        t_start/t_end: Frame range to render.
        fps_out: Output frame rate.
        crop_fraction: Crop window size as fraction of camera frame.
        pip_scale: PiP inset width as fraction of output frame.
        pip_margin_px: PiP margin from frame edge.
        pip_border_px: PiP border thickness.
        annotate: If True, draw camera name and frame index on output.
    """
    # Resolve fps map: explicit arg > module-level CAMERA_FPS > base_fps fallback
    fps_map = camera_fps if camera_fps is not None else (CAMERA_FPS or {})
 
    z = np.load(homography_path)
    H_top_to_cam = {
        cam: z[f"{cam}__H"]
        for cam in side_cameras
        if f"{cam}__H" in z
    }
 
    # Open one frame per camera to get dimensions
    cam_wh = {}
    for cam in side_cameras + ([r.camera for r in rois] if rois else []):
        vp = _build_video_path(video_root, arena, experiment, session, chunk, cam)
        cap = cv2.VideoCapture(str(vp))
        if cap.isOpened():
            cam_wh[cam] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
 
    # Output size: first valid side camera dims
    out_w, out_h = next(iter(cam_wh.values()))
    T = len(selected_cam)
    t_end = t_end if t_end is not None else T - 1
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter failed: {output_path}"
 
    # Keep caps open for speed
    caps = {}
    def get_cap(cam):
        if cam not in caps:
            vp = _build_video_path(video_root, arena, experiment, session, chunk, cam)
            caps[cam] = cv2.VideoCapture(str(vp))
        return caps[cam]
 
    frames_written = 0
    print(f"Rendering frames {t_start}→{t_end} to {output_path.name} ...")
 
    for t in range(t_start, t_end + 1):
        cam = selected_cam[t]
        cap = get_cap(cam)
        cap.set(cv2.CAP_PROP_POS_FRAMES, _cam_frame_idx(t, cam, fps_map, base_fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
 
        w0, h0 = cam_wh.get(cam, (frame.shape[1], frame.shape[0]))
 
        # --- Rotate frame to match East camera orientation ---
        rot = CAMERA_ROTATION.get(cam)
        if rot is not None:
            frame = cv2.rotate(frame, rot)
 
        # After rotation, the canvas dimensions may swap — track separately
        # w0, h0 = original (pre-rotation) dims used for point transform
        # rw, rh = rotated frame dims used for cropping
        if rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rw, rh = h0, w0
        else:
            rw, rh = w0, h0
 
        # --- Crop to follow object ---
        if centers_valid[t] and cam in H_top_to_cam:
            proj = _apply_homography(H_top_to_cam[cam],
                                     centers_top[t:t+1].astype(np.float64))[0]
            # Transform projected point using ORIGINAL dims, then crop using ROTATED dims
            proj = _rotate_point(proj, rot, w0, h0)
            frame = _crop_follow(frame, proj, rw, rh, crop_fraction)
        elif centers_valid[t] and cam not in H_top_to_cam:
            frame = _crop_follow(frame, np.array([rw/2, rh/2]), rw, rh, crop_fraction)
 
        # Resize crop to output size
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
 
        # --- PiP insets for active ROIs ---
        if rois and centers_valid[t]:
            xy = centers_top[t]
            active_rois = [r for r in rois if r.contains(xy)]
            frame = _composite_pips(
                frame, active_rois, t, get_cap, cam_wh,
                out_w, out_h, pip_scale, pip_margin_px, pip_border_px,
                fps_map=fps_map, base_fps=base_fps,
            )
 
        # --- Annotation ---
        if annotate:
            cv2.putText(frame, f"{cam} | t={t}",
                        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2, cv2.LINE_AA)
            if centers_valid[t]:
                cx, cy = centers_top[t]
                # Draw dot at projected object position on the (cropped+resized) frame
                # Note: position is approximate post-crop; dot is drawn at frame center
                # since we cropped around the object
                if centre_dot:
                    cv2.circle(frame, (out_w // 2, out_h // 2), 8, (0, 0, 255), -1)
 
        writer.write(frame)
        frames_written += 1
        if frames_written % 500 == 0:
            print(f"  Written {frames_written} / {t_end - t_start + 1} frames")
 
    writer.release()
    for cap in caps.values():
        cap.release()
 
    print(f"Done. {frames_written} frames → {output_path}")
 
 
def save_rois(rois: list[ROI], path: Path) -> None:
    """Serialise ROI list to JSON."""
    data = []
    for r in rois:
        d = {"name": r.name, "camera": r.camera, "color": list(r.color)}
        if r.center_x is not None:
            d.update({"type": "rect",
                      "center_x": r.center_x, "center_y": r.center_y,
                      "width_px": r.width_px,  "height_px": r.height_px})
        else:
            d.update({"type": "circle", "cx": r.cx, "cy": r.cy, "radius": r.radius})
        data.append(d)
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(rois)} ROIs → {path}")
 
 
def load_rois(path: Path) -> list[ROI]:
    """Load ROIs from JSON."""
    data = json.loads(path.read_text())
    rois = []
    for d in data:
        color = tuple(d.get("color", [255, 255, 255]))
        if d["type"] == "rect":
            rois.append(ROI.rect(d["name"], d["camera"],
                                 d["center_x"], d["center_y"],
                                 d["width_px"], d["height_px"], color))
        else:
            rois.append(ROI.circle(d["name"], d["camera"], d["cx"], d["cy"], d["radius"], color))
    return rois
 
 
# --- Under the hood -----------------------------------------------------------
 
def _cam_frame_idx(t: int, camera: str, fps_map: dict[str, float], base_fps: float = BASE_FPS) -> int:
    """
    Convert a base-fps frame index to the correct frame index for a
    camera that may run at a different fps (e.g. Patch cameras at 100fps).
 
    Example: t=1000 at 50fps base, Patch camera at 100fps → frame 2000.
    """
    ratio = fps_map.get(camera, base_fps) / base_fps
    return int(round(t * ratio))
 
 
def _apply_homography(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    ones = np.ones((xy.shape[0], 1))
    X = np.concatenate([xy, ones], axis=1)
    Y = (H @ X.T).T
    return Y[:, :2] / Y[:, 2:3]
 
 
def _in_bounds(p, w, h, margin=0.0) -> bool:
    x, y = float(p[0]), float(p[1])
    return (margin <= x <= w - 1 - margin) and (margin <= y <= h - 1 - margin)
 
 
def _center_distance_norm(p, w, h) -> float:
    dx = (float(p[0]) - w / 2) / (w / 2)
    dy = (float(p[1]) - h / 2) / (h / 2)
    return float(np.hypot(dx, dy))
 
 
def _side_of_line(pts: np.ndarray, p1: tuple, p2: tuple) -> np.ndarray:
    """
    Returns sign of cross product (p2-p1) × (pt-p1) for each point.
    Positive = left, negative = right.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (pts[:, 0] - p1[0]) * dy - (pts[:, 1] - p1[1]) * dx
 
 
def _lookup_best_cam(xy, best_cam_idx, xs, ys, side_cameras, grid_step_px) -> str | None:
    x, y = float(xy[0]), float(xy[1])
    xi = int(np.clip(np.round(x / grid_step_px), 0, len(xs) - 1))
    yi = int(np.clip(np.round(y / grid_step_px), 0, len(ys) - 1))
    ci = int(best_cam_idx[yi, xi])
    return side_cameras[ci] if ci >= 0 else None
 
 
def _interpolate_centers(
    centers: np.ndarray, valid: np.ndarray, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation over gaps shorter than max_gap frames."""
    centers = centers.copy()
    new_valid = valid.copy()
    T = len(centers)
    i = 0
    while i < T:
        if not valid[i]:
            i += 1
            continue
        # find next gap
        j = i + 1
        while j < T and valid[j]:
            j += 1
        if j >= T:
            break
        # j is start of gap — find end
        k = j
        while k < T and not valid[k]:
            k += 1
        if k >= T:
            break
        gap = k - j
        if gap <= max_gap:
            for g in range(gap):
                alpha = (g + 1) / (gap + 1)
                centers[j + g] = (1 - alpha) * centers[i] + alpha * centers[k]
                new_valid[j + g] = True
        i = k
    return centers, new_valid
 
 
def _crop_follow(
    frame: np.ndarray,
    center_xy: np.ndarray,
    frame_w: int,
    frame_h: int,
    crop_fraction: float,
) -> np.ndarray:
    """
    Crop a window of size (crop_fraction * W, crop_fraction * H) around
    center_xy, clamped to frame boundaries.
    """
    crop_w = int(frame_w * crop_fraction)
    crop_h = int(frame_h * crop_fraction)
 
    cx, cy = int(np.clip(center_xy[0], 0, frame_w - 1)), \
             int(np.clip(center_xy[1], 0, frame_h - 1))
 
    x0 = int(np.clip(cx - crop_w // 2, 0, frame_w - crop_w))
    y0 = int(np.clip(cy - crop_h // 2, 0, frame_h - crop_h))
 
    return frame[y0:y0 + crop_h, x0:x0 + crop_w]
 
 
def _composite_pips(
    frame: np.ndarray,
    active_rois: list[ROI],
    t: int,
    get_cap,
    cam_wh: dict,
    out_w: int,
    out_h: int,
    pip_scale: float,
    margin: int,
    border: int,
    fps_map: dict[str, float] | None = None,
    base_fps: float = BASE_FPS,
) -> np.ndarray:
    """
    Composite PiP insets for each active ROI into the frame.
    Insets are stacked from the top-right corner downward.
    """
    pip_w = int(out_w * pip_scale)
 
    corners = [
        (out_w - pip_w - margin, margin),           # top-right
        (margin, margin),                            # top-left
        (out_w - pip_w - margin, out_h // 2),        # mid-right
        (margin, out_h // 2),                        # mid-left
    ]
 
    for idx, roi in enumerate(active_rois):
        cap = get_cap(roi.camera)
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                _cam_frame_idx(t, roi.camera, fps_map or {}, base_fps))
        ok, pip_frame = cap.read()
        if not ok or pip_frame is None:
            continue
 
        # Scale inset preserving aspect ratio
        rw, rh = cam_wh.get(roi.camera, (pip_frame.shape[1], pip_frame.shape[0]))
        pip_h = int(pip_w * rh / rw)
        pip_frame = cv2.resize(pip_frame, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
 
        # Border
        pip_frame = cv2.copyMakeBorder(
            pip_frame, border, border, border, border,
            cv2.BORDER_CONSTANT, value=list(roi.color)
        )
        pip_h_b = pip_frame.shape[0]
        pip_w_b = pip_frame.shape[1]
 
        if idx >= len(corners):
            break
        ox, oy = corners[idx]
 
        # Clamp to frame
        oy_end = min(oy + pip_h_b, out_h)
        ox_end = min(ox + pip_w_b, out_w)
        pip_crop = pip_frame[:oy_end - oy, :ox_end - ox]
        frame[oy:oy_end, ox:ox_end] = pip_crop
 
        # Label
        cv2.putText(frame, roi.name, (ox + border + 4, oy + border + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(int(c) for c in roi.color),
                    2, cv2.LINE_AA)
 
    return frame
 
 
def _rotate_point(
    pt: np.ndarray,
    rot,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Rotate a point (x, y) to match a cv2.rotate() applied to a (w, h) frame.
    w and h are the ORIGINAL frame dimensions (before rotation).
    Returns the transformed point in the rotated frame's coordinate system.
    """
    if rot is None:
        return pt
    x, y = float(pt[0]), float(pt[1])
    if rot == cv2.ROTATE_90_CLOCKWISE:
        return np.array([h - 1 - y, x], dtype=np.float64)
    if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return np.array([y, w - 1 - x], dtype=np.float64)
    if rot == cv2.ROTATE_180:
        return np.array([w - 1 - x, h - 1 - y], dtype=np.float64)
    return pt
 
 
def _build_video_path(root, arena, experiment, session, chunk, camera) -> Path:
    return Path(root) / arena / experiment / session / camera / f"{camera}_{chunk}.avi"
 
 
# --- Main (example usage — see notebook for interactive ROI definition) -------
 
if __name__ == "__main__":
    print("Import this module and use functions interactively.")
    print("See notebook for ROI definition and preference map visualisation.")
 