from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np


@dataclass
class GestureResult:
    """High-level gesture command inferred from a single frame."""

    command: str
    confidence: float
    static_label: str
    contour_area: float
    centroid: tuple[int, int] | None
    motion_px: float
    bbox: tuple[int, int, int, int] | None
    mask_ratio: float
    hu_distance: float
    swipe_score: float
    motion_angle_deg: float | None
    stable_frames: int


class GestureRecognizer:
    """
    Gesture recognition pipeline aligned with the course slides.

    Static gesture classification:
        - Segment the hand
        - Compute similitude-invariant Hu moments
        - Compare against template hand shapes

    Dynamic gesture classification:
        - Maintain a Motion History Image (MHI) from frame differencing
        - Detect a rapid horizontal swipe to trigger clear
    """

    def __init__(
        self,
        min_contour_area: int = 3500,
        smooth_window: int = 5,
        mhi_duration: int = 12,
        swipe_cooldown: int = 10,
        template_path: str | None = None,
        command_hold_frames: int = 3,
    ) -> None:
        self.min_contour_area = min_contour_area
        self.smooth_window = smooth_window
        self.mhi_duration = mhi_duration
        self.swipe_cooldown = swipe_cooldown
        self.command_hold_frames = command_hold_frames

        self._labels: deque[str] = deque(maxlen=smooth_window)
        self._confidences: deque[float] = deque(maxlen=smooth_window)
        self._prev_centroid: tuple[int, int] | None = None
        self._prev_gray: np.ndarray | None = None
        self._mhi: np.ndarray | None = None
        self._frame_index = 0
        self._cooldown = 0
        self._centroid_history: deque[tuple[int, int]] = deque(maxlen=max(6, smooth_window + 2))
        self._stable_command = "idle"
        self._stable_frames = 0
        self._candidate_command = "idle"
        self._candidate_frames = 0
        self._template_path = Path(template_path) if template_path else None
        self._shape_templates = self._build_shape_templates(self._template_path)

    def reset(self) -> None:
        """Clear temporal state between video sessions."""
        self._labels.clear()
        self._confidences.clear()
        self._prev_centroid = None
        self._prev_gray = None
        self._mhi = None
        self._frame_index = 0
        self._cooldown = 0
        self._centroid_history.clear()
        self._stable_command = "idle"
        self._stable_frames = 0
        self._candidate_command = "idle"
        self._candidate_frames = 0

    def update(self, frame: np.ndarray) -> GestureResult:
        """Process one BGR frame and return the current gesture command."""
        self._frame_index += 1
        mask = self._skin_mask(frame)
        contour = self._largest_contour(mask)
        swipe_score, motion_angle = self._update_mhi(frame)
        clear_triggered = self._detect_swipe_clear(swipe_score, motion_angle)

        if contour is None:
            self._labels.append("clear" if clear_triggered else "idle")
            self._confidences.append(0.98 if clear_triggered else 0.0)
            return GestureResult(
                command="clear" if clear_triggered else "idle",
                confidence=0.98 if clear_triggered else 0.0,
                static_label="none",
                contour_area=0.0,
                centroid=None,
                motion_px=0.0,
                bbox=None,
                mask_ratio=float(mask.mean() / 255.0),
                hu_distance=float("inf"),
                swipe_score=swipe_score,
                motion_angle_deg=motion_angle,
                stable_frames=self._stable_frames,
            )

        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        centroid = self._centroid(contour)
        motion_px = 0.0
        if centroid is not None and self._prev_centroid is not None:
            motion_px = float(np.linalg.norm(np.subtract(centroid, self._prev_centroid)))
        self._prev_centroid = centroid
        if centroid is not None:
            self._centroid_history.append(centroid)

        if area < self.min_contour_area:
            self._labels.append("clear" if clear_triggered else "idle")
            self._confidences.append(0.98 if clear_triggered else 0.1)
            return GestureResult(
                command="clear" if clear_triggered else "idle",
                confidence=0.98 if clear_triggered else 0.1,
                static_label="noise",
                contour_area=area,
                centroid=centroid,
                motion_px=motion_px,
                bbox=(int(x), int(y), int(w), int(h)),
                mask_ratio=float(mask.mean() / 255.0),
                hu_distance=float("inf"),
                swipe_score=swipe_score,
                motion_angle_deg=motion_angle,
                stable_frames=self._stable_frames,
            )

        static_label, hu_distance = self._classify_static_gesture(contour, frame.shape[:2])
        command, confidence = self._command_from_slide_methods(
            static_label=static_label,
            hu_distance=hu_distance,
            swipe_score=swipe_score,
            clear_triggered=clear_triggered,
        )
        self._labels.append(command)
        self._confidences.append(confidence)
        smoothed = self._stabilize_command(command, confidence, clear_triggered)

        return GestureResult(
            command=smoothed,
            confidence=0.98 if smoothed == "clear" else confidence,
            static_label=static_label,
            contour_area=area,
            centroid=centroid,
            motion_px=motion_px,
            bbox=(int(x), int(y), int(w), int(h)),
            mask_ratio=float(mask.mean() / 255.0),
            hu_distance=hu_distance,
            swipe_score=swipe_score,
            motion_angle_deg=motion_angle,
            stable_frames=self._stable_frames,
        )

    def _skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """Build a conservative hand mask from YCrCb and HSV thresholds."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        mask_hsv = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
        mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _largest_contour(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _centroid(self, contour: np.ndarray) -> tuple[int, int] | None:
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            return None
        return (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )

    def _classify_static_gesture(
        self,
        contour: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> tuple[str, float]:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        hu = self._hu_descriptor(mask)
        best_label = "unknown"
        best_distance = float("inf")
        for label, template_hu in self._shape_templates.items():
            dist = float(np.mean(np.abs(hu - template_hu)))
            if dist < best_distance:
                best_label = label
                best_distance = dist
        if best_distance > 0.95:
            return "unknown", best_distance
        return best_label, best_distance

    def _hu_descriptor(self, mask: np.ndarray) -> np.ndarray:
        moments = cv2.moments(mask)
        hu = cv2.HuMoments(moments).flatten()
        return -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    def _build_shape_templates(self, template_path: Path | None = None) -> dict[str, np.ndarray]:
        size = 160
        templates = {
            "point": self._hu_descriptor(self._template_mask(size, "point")),
            "open_palm": self._hu_descriptor(self._template_mask(size, "open_palm")),
            "fist": self._hu_descriptor(self._template_mask(size, "fist")),
        }
        if template_path is not None and template_path.exists():
            try:
                loaded = json.loads(template_path.read_text())
                for key, value in loaded.items():
                    arr = np.array(value, dtype=np.float64)
                    if arr.shape == (7,):
                        templates[key] = arr
            except Exception:
                pass
        return templates

    def save_templates(self, path: str, templates: dict[str, np.ndarray]) -> None:
        """Persist user-calibrated Hu templates as JSON."""
        payload = {key: np.asarray(value, dtype=np.float64).tolist() for key, value in templates.items()}
        Path(path).write_text(json.dumps(payload, indent=2))
        self._shape_templates = self._build_shape_templates(Path(path))

    def _template_mask(self, size: int, label: str) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.uint8)
        if label == "point":
            cv2.rectangle(mask, (56, 78), (104, 138), 255, -1)
            cv2.rectangle(mask, (72, 24), (88, 94), 255, -1)
            cv2.circle(mask, (80, 20), 10, 255, -1)
        elif label == "open_palm":
            cv2.rectangle(mask, (48, 70), (112, 132), 255, -1)
            finger_specs = [(46, 24, 58, 82), (62, 16, 74, 82), (78, 14, 90, 82), (94, 18, 106, 82), (108, 30, 122, 94)]
            for x0, y0, x1, y1 in finger_specs:
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        elif label == "fist":
            cv2.rectangle(mask, (48, 58), (112, 126), 255, -1)
            cv2.circle(mask, (80, 58), 32, 255, -1)
        else:
            raise ValueError(f"Unknown template label: {label}")
        return mask

    def _update_mhi(self, frame: np.ndarray) -> tuple[float, float | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._mhi is None:
            self._mhi = np.zeros_like(gray, dtype=np.float32)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0, None

        diff = cv2.absdiff(gray, self._prev_gray)
        _, silhouette = cv2.threshold(diff, 22, 1, cv2.THRESH_BINARY)
        self._mhi = np.maximum(self._mhi - 1.0, 0.0)
        self._mhi[silhouette > 0] = float(self.mhi_duration)
        self._prev_gray = gray

        recent_motion = (self._mhi >= float(self.mhi_duration) * 0.45).astype(np.uint8) * 255
        contours, _ = cv2.findContours(recent_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        frame_area = float(recent_motion.shape[0] * recent_motion.shape[1])
        if area < 0.004 * frame_area:
            return area / frame_area, None

        angle = self._principal_axis_angle(contour)
        return area / frame_area, angle

    def _principal_axis_angle(self, contour: np.ndarray) -> float | None:
        moments = cv2.moments(contour)
        mu20 = moments["mu20"]
        mu02 = moments["mu02"]
        mu11 = moments["mu11"]
        denom = mu20 - mu02
        if abs(denom) < 1e-6 and abs(mu11) < 1e-6:
            return None
        angle = 0.5 * np.degrees(np.arctan2(2.0 * mu11, denom))
        return float(angle)

    def _detect_swipe_clear(self, swipe_score: float, motion_angle: float | None) -> bool:
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if len(self._centroid_history) < 4:
            return False

        start = np.array(self._centroid_history[0], dtype=np.float32)
        end = np.array(self._centroid_history[-1], dtype=np.float32)
        dx, dy = end - start
        travel = float(np.linalg.norm(end - start))
        horizontal = abs(dx) > max(35.0, 1.8 * abs(dy))
        orientation_ok = motion_angle is None or abs(motion_angle) < 30.0 or abs(abs(motion_angle) - 180.0) < 30.0
        fast_enough = travel > 90.0
        enough_motion = swipe_score > 0.015
        if horizontal and orientation_ok and fast_enough and enough_motion:
            self._cooldown = self.swipe_cooldown
            self._centroid_history.clear()
            return True
        return False

    def _command_from_slide_methods(
        self,
        *,
        static_label: str,
        hu_distance: float,
        swipe_score: float,
        clear_triggered: bool,
    ) -> tuple[str, float]:
        if clear_triggered:
            return "clear", 0.98
        if static_label == "point":
            confidence = float(np.clip(1.0 - hu_distance / 1.2, 0.55, 0.95))
            return "draw", confidence
        if static_label == "open_palm":
            confidence = float(np.clip(1.0 - hu_distance / 1.1, 0.55, 0.95))
            return "erase", confidence
        if static_label == "fist" and swipe_score < 0.01:
            return "idle", 0.45
        return "idle", 0.2

    def _stabilize_command(self, command: str, confidence: float, clear_triggered: bool) -> str:
        if clear_triggered:
            self._stable_command = "clear"
            self._stable_frames = 1
            self._candidate_command = "idle"
            self._candidate_frames = 0
            return "clear"

        if command == self._stable_command:
            self._stable_frames += 1
            self._candidate_command = command
            self._candidate_frames = 0
            return self._stable_command

        if confidence < 0.45:
            self._candidate_command = "idle"
            self._candidate_frames = 0
            return self._stable_command

        if command == self._candidate_command:
            self._candidate_frames += 1
        else:
            self._candidate_command = command
            self._candidate_frames = 1

        if self._candidate_frames >= self.command_hold_frames:
            self._stable_command = self._candidate_command
            self._stable_frames = self._candidate_frames
            self._candidate_frames = 0
        return self._stable_command

    def contour_to_hu(self, contour: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        return self._hu_descriptor(mask)


def annotate_gesture(frame: np.ndarray, result: GestureResult) -> np.ndarray:
    """Draw a compact QA overlay for the current gesture result."""
    vis = frame.copy()
    if result.bbox is not None:
        x, y, w, h = result.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    if result.centroid is not None:
        cv2.circle(vis, result.centroid, 6, (0, 255, 0), -1)

    label = f"{result.command.upper()}  static={result.static_label}  conf={result.confidence:.2f}"
    details = f"hu={result.hu_distance:.2f}  swipe={result.swipe_score:.3f}  motion={result.motion_px:.1f}px"
    angle_txt = "n/a" if result.motion_angle_deg is None else f"{result.motion_angle_deg:.1f}deg"
    details2 = f"area={result.contour_area:.0f}  angle={angle_txt}  stable={result.stable_frames}"
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 220, 30), 2)
    cv2.putText(vis, details, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.putText(vis, details2, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    return vis
