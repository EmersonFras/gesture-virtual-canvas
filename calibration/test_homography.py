"""Unit tests for DLT homography and apply_homography."""
import numpy as np
import sys
sys.path.insert(0, '.')

from calibration.homography import compute_homography_dlt, apply_homography


def test_identity_like():
    """Points that map to themselves should give near-identity H."""
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    dst = src.copy()
    H = compute_homography_dlt(src, dst)
    for s, d in zip(src, dst):
        proj = apply_homography(H, tuple(s))
        err = np.linalg.norm(np.array(proj) - d)
        assert err < 0.5, f"Identity test failed: err={err:.4f}"
    print("PASS: identity-like homography")


def test_reprojection_canvas():
    """Simulated image corners → canvas corners reprojection error < 2px."""
    src = np.array([[120, 80], [520, 90], [510, 400], [130, 390]], dtype=np.float64)
    canvas_w, canvas_h = 640, 480
    dst = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], dtype=np.float64)
    H = compute_homography_dlt(src, dst)

    assert np.linalg.det(H) > 0, "det(H) must be positive"
    assert np.linalg.cond(H) < 1e6, "H is poorly conditioned"

    for s, d in zip(src, dst):
        proj = apply_homography(H, tuple(s))
        err = np.linalg.norm(np.array(proj) - d)
        assert err < 2.0, f"Reprojection error too large: {err:.2f}px  src={s} expected={d} got={proj}"
    print("PASS: canvas reprojection < 2px")


def test_roundtrip():
    """Applying H then H_inv should recover original point within 1px."""
    src = np.array([[50, 40], [600, 45], [590, 440], [55, 435]], dtype=np.float64)
    dst = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], dtype=np.float64)
    H = compute_homography_dlt(src, dst)
    H_inv = np.linalg.inv(H)
    pt = (300.0, 200.0)
    canvas_pt = apply_homography(H, pt)
    recovered = apply_homography(H_inv, canvas_pt)
    err = np.linalg.norm(np.array(recovered) - np.array(pt))
    assert err < 1.0, f"Roundtrip error: {err:.4f}px"
    print("PASS: roundtrip H -> H_inv < 1px")


if __name__ == '__main__':
    test_identity_like()
    test_reprojection_canvas()
    test_roundtrip()
    print("\nAll tests passed.")
