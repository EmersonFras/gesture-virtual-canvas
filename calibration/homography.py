import numpy as np


def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize points so centroid is at origin and mean distance is sqrt(2)."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-8:
        mean_dist = 1.0
    scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0,     -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,     0,      1.0               ],
    ])
    ones = np.ones((len(pts), 1))
    pts_h = np.hstack([pts, ones])
    pts_n = (T @ pts_h.T).T
    return pts_n[:, :2], T


def compute_homography_dlt(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 homography from 4 point correspondences using normalized DLT.

    Args:
        src_pts: (4, 2) source points in image space
        dst_pts: (4, 2) destination points in canvas space

    Returns:
        H: (3, 3) homography matrix mapping src → dst
    """
    src_n, T_src = _normalize_points(src_pts)
    dst_n, T_dst = _normalize_points(dst_pts)

    A = []
    for (x, y), (xp, yp) in zip(src_n, dst_n):
        A.append([-x, -y, -1,  0,   0,  0,  xp * x,  xp * y,  xp])
        A.append([ 0,  0,  0, -x,  -y, -1,  yp * x,  yp * y,  yp])
    A = np.array(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_n = h.reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_n @ T_src
    H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    """
    Apply a 3x3 homography to a single 2D point.

    Args:
        H:  (3, 3) homography matrix
        pt: (x, y) in source coordinate space

    Returns:
        (x', y') in destination coordinate space
    """
    x, y = pt
    p = np.array([x, y, 1.0], dtype=np.float64)
    pp = H @ p
    return (pp[0] / pp[2], pp[1] / pp[2])
