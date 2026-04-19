import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as [TL, TR, BR, BL].

    Uses the centroid-angle method: sort CCW by angle from centroid, then
    rotate so the point with smallest x+y (closest to image top-left) is first.

    Args:
        pts: (4, 2) array of (x, y) corner points in any order

    Returns:
        (4, 2) float32 array ordered [TL, TR, BR, BL]
    """
    pts = np.array(pts, dtype=np.float32)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    pts = pts[order]

    sums = pts[:, 0] + pts[:, 1]
    tl_idx = int(np.argmin(sums))
    pts = np.roll(pts, -tl_idx, axis=0)

    tl, tr, br, bl = pts
    # Ensure CCW winding if not, swap tr and bl
    if np.cross(tr - tl, bl - tl) < 0:
        pts = np.array([tl, bl, br, tr], dtype=np.float32)

    return pts.astype(np.float32)


def aspect_ratio_ok(corners: np.ndarray, tolerance: float = 0.20) -> bool:
    """
    Check that the quadrilateral aspect ratio matches A4 or Letter within tolerance.

    A4  ~1.414, Letter ~1.294
    """
    tl, tr, br, bl = corners
    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    w = (top_w + bot_w) / 2
    h = (left_h + right_h) / 2
    if h < 1e-3:
        return False
    ratio = max(w, h) / min(w, h)
    for target in (1.414, 1.294):
        if abs(ratio - target) / target <= tolerance:
            return True
    return False
