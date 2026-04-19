import numpy as np
from collections import deque

from calibration.canny import canny_edges
from calibration.harris import harris_corners, refine_corners
from calibration.corners import order_corners, aspect_ratio_ok
from calibration.homography import compute_homography_dlt, apply_homography

# Canvas output height — width derived from detected aspect ratio
CANVAS_H = 600
# Minimum paper area as fraction of total frame area
MIN_AREA_RATIO = 0.05
ASPECT_TOL = 0.20


def _dilate(binary: np.ndarray, radius: int = 3) -> np.ndarray:
    """Binary dilation with a square kernel via array slicing"""
    result = np.zeros_like(binary, dtype=bool)
    H, W = binary.shape
    b = binary.astype(bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r0, r1 = max(0, dr),  min(H, H + dr)
            c0, c1 = max(0, dc),  min(W, W + dc)
            sr0, sr1 = max(0, -dr), min(H, H - dr)
            sc0, sc1 = max(0, -dc), min(W, W - dc)
            result[r0:r1, c0:c1] |= b[sr0:sr1, sc0:sc1]
    return result


def _paper_quad(edges: np.ndarray) -> np.ndarray | None:
    """
    Find 4 paper corner anchors by flood-filling the enclosed paper region.

    Dilates edges to close corner gaps, then BFS from the frame center through
    non-edge pixels to find the paper interior. Extremal points of that region
    give TL/TR/BR/BL anchors for Harris refinement.
    Returns (4, 2) float32 array of (x, y) positions, or None.
    """
    closed = _dilate(edges, radius=3)

    H, W = closed.shape
    start = (H // 2, W // 2)
    if closed[start]:
        return None

    visited = np.zeros((H, W), dtype=bool)
    visited[start] = True
    queue = deque([start])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and not closed[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    pts = np.argwhere(visited)
    if len(pts) < 4:
        return None

    xy = pts[:, ::-1].astype(np.float32)  # (N,2) as (x,y)
    sums = xy[:, 0] + xy[:, 1]
    diffs = xy[:, 0] - xy[:, 1]

    tl = xy[np.argmin(sums)]
    br = xy[np.argmax(sums)]
    tr = xy[np.argmax(diffs)]
    bl = xy[np.argmin(diffs)]

    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < H * W * MIN_AREA_RATIO:
        return None

    return quad


def _canvas_size(corners: np.ndarray) -> tuple[int, int]:
    """Derive canvas pixel dimensions from detected corner geometry."""
    tl, tr, br, bl = corners
    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    w_avg = (top_w + bot_w) / 2
    h_avg = (left_h + right_h) / 2
    ratio = w_avg / max(h_avg, 1.0)
    canvas_h = CANVAS_H
    canvas_w = int(round(canvas_h * ratio))
    return canvas_w, canvas_h


def detect_canvas(frame: np.ndarray) -> dict | None:
    """
    Detect a white paper sheet on a desk and return a homography to its canvas space.

    Args:
        frame: BGR image from camera, shape (H, W, 3), dtype uint8

    Returns:
        None if no valid paper canvas detected, otherwise a dict with keys:
            H           (3,3) float64 — image coords → canvas coords
            H_inv       (3,3) float64 — canvas coords → image coords
            canvas_w    int
            canvas_h    int
            corners_img (4,2) float32 — [TL, TR, BR, BL] in image space
    """
    edges, Ix, Iy = canny_edges(frame)

    approx = _paper_quad(edges)
    if approx is None:
        return None

    approx_ordered = order_corners(approx)
    if not aspect_ratio_ok(approx_ordered, tolerance=ASPECT_TOL):
        return None

    harris_pts = harris_corners(Ix, Iy)
    refined = refine_corners(approx_ordered, harris_pts, search_radius=20)
    corners = refined if refined is not None else approx_ordered
    corners = order_corners(corners)

    if not aspect_ratio_ok(corners, tolerance=ASPECT_TOL):
        return None

    canvas_w, canvas_h = _canvas_size(corners)

    src_pts = corners.astype(np.float64)
    dst_pts = np.array([
        [0,        0       ],
        [canvas_w, 0       ],
        [canvas_w, canvas_h],
        [0,        canvas_h],
    ], dtype=np.float64)

    H = compute_homography_dlt(src_pts, dst_pts)

    if np.linalg.det(H) <= 0:
        return None
    if np.linalg.cond(H) > 1e6:
        return None

    H_inv = np.linalg.inv(H)

    return {
        "H":           H,
        "H_inv":       H_inv,
        "canvas_w":    canvas_w,
        "canvas_h":    canvas_h,
        "corners_img": corners,
    }
