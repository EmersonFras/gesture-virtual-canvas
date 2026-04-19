# Integration Guide

Detects a white paper sheet in a camera frame and returns a homography mapping
image pixel coordinates to a flat canvas coordinate space (origin at paper top-left).

## API

### `detect_canvas(frame) -> dict | None`

```python
from calibration.detector import detect_canvas

result = detect_canvas(frame)   # frame: RGB uint8 ndarray from cv2
```

Returns `None` if no paper is detected. Otherwise:

| Key | Type | Description |
|-----|------|-------------|
| `H` | `(3,3) float64` | Image coords -> canvas coords |
| `H_inv` | `(3,3) float64` | Canvas coords -> image coords |
| `canvas_w` | `int` | Canvas width in pixels |
| `canvas_h` | `int` | Canvas height in pixels (always 600) |
| `corners_img` | `(4,2) float32` | `[TL, TR, BR, BL]` in image space |

### `apply_homography(H, pt) -> (float, float)`

```python
from calibration.homography import apply_homography

canvas_xy = apply_homography(result["H"],     (img_x,    img_y))
img_xy    = apply_homography(result["H_inv"], (canvas_x, canvas_y))
```

## Performance

Cache the result of `detect_canvas` and refresh every 15–30 frames rather than
calling it on every frame.

## QA Tool

```bash
python calibration/qa_visualize.py --image path/to/image.jpg
python calibration/qa_visualize.py --camera 0
python calibration/qa_visualize.py --video path/to/video.mp4
```

Keys: `q` = quit, `s` = save frame to `qa_output/`.
