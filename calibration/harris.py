import numpy as np
from calibration.canny import _gaussian_kernel, _convolve2d


def harris_corners(
    Ix: np.ndarray,
    Iy: np.ndarray,
    k: float = 0.05,
    sigma: float = 1.5,
    threshold_ratio: float = 0.01,
    nms_radius: int = 7,
) -> list[tuple[int, int, float]]:
    """
    Compute Harris corner response and return list of (row, col, score).

    Args:
        Ix, Iy:          Sobel gradients from canny step
        k:               Harris sensitivity parameter
        sigma:           Gaussian window sigma for structure tensor smoothing
        threshold_ratio: keep pixels with R > threshold_ratio * max(R)
        nms_radius:      non-max suppression neighborhood half-size

    Returns:
        List of (row, col, score) sorted descending by score
    """
    A = Ix * Ix
    B = Ix * Iy
    C = Iy * Iy

    kern_k = max(1, int(2 * sigma))
    kernel = _gaussian_kernel(kern_k, sigma)
    A_bar = _convolve2d(A, kernel)
    B_bar = _convolve2d(B, kernel)
    C_bar = _convolve2d(C, kernel)

    det_M = A_bar * C_bar - B_bar ** 2
    trace_M = A_bar + C_bar
    R = det_M - k * trace_M ** 2

    threshold = threshold_ratio * R.max()
    candidates = np.argwhere(R > threshold)

    # NMS: sort by score, suppress neighbors
    scores = R[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-scores)
    candidates = candidates[order]
    scores = scores[order]

    kept = []
    suppressed = np.zeros(R.shape, dtype=bool)
    for (r, c), score in zip(candidates, scores):
        if suppressed[r, c]:
            continue
        kept.append((r, c, float(score)))
        r0 = max(0, r - nms_radius)
        r1 = min(R.shape[0], r + nms_radius + 1)
        c0 = max(0, c - nms_radius)
        c1 = min(R.shape[1], c + nms_radius + 1)
        suppressed[r0:r1, c0:c1] = True

    return kept


def refine_corners(
    approx_corners: np.ndarray,
    harris_pts: list[tuple[int, int, float]],
    search_radius: int = 20,
) -> np.ndarray:
    """
    For each approximate corner, find the highest-score Harris point within search_radius.

    Args:
        approx_corners: (4, 2) float32 array of approximate (x, y) corner locations
        harris_pts:     list of (row, col, score) from harris_corners()
        search_radius:  pixel radius to search around each approximate corner

    Returns:
        (4, 2) float32 refined corners, or None if any corner has no nearby Harris point
    """
    if not harris_pts:
        return None

    harris_arr = np.array([[c, r] for r, c, _ in harris_pts], dtype=np.float32)  # (N,2) as (x,y)
    refined = np.zeros((4, 2), dtype=np.float32)

    for i, (ax, ay) in enumerate(approx_corners):
        dists = np.sqrt(((harris_arr - np.array([ax, ay])) ** 2).sum(axis=1))
        mask = dists <= search_radius
        if not mask.any():
            return None
        best = np.argmin(dists * (~mask * 1e9 + 1))
        refined[i] = harris_arr[best]

    return refined
