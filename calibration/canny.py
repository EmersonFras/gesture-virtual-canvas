import numpy as np
from collections import deque


def _gaussian_kernel(k: int, sigma: float) -> np.ndarray:
    size = 2 * k + 1
    ax = np.arange(-k, k + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution via separable 1D passes."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def rgb_to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale."""
    return (0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0])


def gaussian_blur(gray: np.ndarray, k: int = 2, sigma: float = 1.0) -> np.ndarray:
    kernel = _gaussian_kernel(k, sigma)
    return _convolve2d(gray, kernel)


def _sobel_gradients(blurred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    Ix = _convolve2d(blurred, Kx)
    Iy = _convolve2d(blurred, Ky)
    return Ix, Iy


def _non_max_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Vectorized NMS, no Python pixel loops."""
    angle_deg = np.degrees(angle) % 180
    padded = np.pad(mag, 1, mode='constant', constant_values=0)

    # Four direction masks
    m0   = ((angle_deg >= 0)     & (angle_deg < 22.5))  | \
           ((angle_deg >= 157.5) & (angle_deg <= 180))
    m45  =  (angle_deg >= 22.5)  & (angle_deg < 67.5)
    m90  =  (angle_deg >= 67.5)  & (angle_deg < 112.5)
    m135 =  (angle_deg >= 112.5) & (angle_deg < 157.5)

    # Neighbor pairs for each direction
    n1_0,   n2_0   = padded[1:-1, :-2],  padded[1:-1, 2:]   # left / right
    n1_45,  n2_45  = padded[:-2,  2:],   padded[2:,  :-2]   # top-right / bot-left
    n1_90,  n2_90  = padded[:-2,  1:-1], padded[2:,  1:-1]  # top / bottom
    n1_135, n2_135 = padded[:-2,  :-2],  padded[2:,  2:]    # top-left / bot-right

    keep = (
        (m0   & (mag >= n1_0)   & (mag >= n2_0))   |
        (m45  & (mag >= n1_45)  & (mag >= n2_45))  |
        (m90  & (mag >= n1_90)  & (mag >= n2_90))  |
        (m135 & (mag >= n1_135) & (mag >= n2_135))
    )
    suppressed = np.where(keep, mag, 0.0)
    return suppressed


def _hysteresis(suppressed: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    H, W = suppressed.shape
    strong = suppressed >= t_high
    weak = (suppressed >= t_low) & ~strong
    output = np.zeros((H, W), dtype=np.uint8)
    output[strong] = 255

    queue = deque(zip(*np.where(strong)))
    visited = strong.copy()
    dirs = [(-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and weak[nr, nc]:
                visited[nr, nc] = True
                output[nr, nc] = 255
                queue.append((nr, nc))
    return output


def canny_edges(
    frame: np.ndarray,
    k: int = 2,
    sigma: float = 1.0,
    t_low_ratio: float = 0.05,
    t_high_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Canny edge detection pipeline from scratch.

    Args:
        frame:        RGB image
        k:            Gaussian half-size (kernel is (2k+1)x(2k+1))
        sigma:        Gaussian sigma
        t_low_ratio:  low threshold as fraction of max magnitude
        t_high_ratio: high threshold as fraction of max magnitude

    Returns:
        edges:   (H, W) uint8 binary edge map (0 or 255)
        Ix:      (H, W) float64 x-gradient (reused by Harris)
        Iy:      (H, W) float64 y-gradient (reused by Harris)
    """
    gray = rgb_to_gray(frame)
    blurred = gaussian_blur(gray, k=k, sigma=sigma)
    Ix, Iy = _sobel_gradients(blurred)
    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    angle = np.arctan2(Iy, Ix)
    suppressed = _non_max_suppression(mag, angle)
    max_m = suppressed.max()
    t_low = t_low_ratio * max_m
    t_high = t_high_ratio * max_m
    edges = _hysteresis(suppressed, t_low, t_high)
    return edges, Ix, Iy
