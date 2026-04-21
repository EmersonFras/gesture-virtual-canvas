"""
Capture user-specific Hu-moment templates for smoother gesture recognition.

Usage:
    python calibration/gesture_calibrate.py
    python calibration/gesture_calibrate.py --camera 1 --output calibration/gesture_templates.json

Keys:
    p = capture current hand as point
    o = capture current hand as open_palm
    f = capture current hand as fist
    r = reset collected samples
    s = save averaged templates
    q = quit
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, '.')

from calibration.gesture import GestureRecognizer, annotate_gesture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    parser.add_argument('--output', default='calibration/gesture_templates.json', help='Output JSON path')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open camera {args.camera}')

    recognizer = GestureRecognizer(smooth_window=3, command_hold_frames=2)
    samples: dict[str, list[np.ndarray]] = defaultdict(list)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        half_w = frame.shape[1] // 2
        gesture_frame = frame[:, :half_w]
        result = recognizer.update(gesture_frame)
        vis = annotate_gesture(frame, result)
        cv2.line(vis, (half_w, 0), (half_w, vis.shape[0]), (100, 100, 100), 1)

        y = 110
        instructions = [
            'p=point  o=open_palm  f=fist',
            's=save templates  r=reset  q=quit',
            f"counts point={len(samples['point'])} palm={len(samples['open_palm'])} fist={len(samples['fist'])}",
        ]
        for line in instructions:
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 2)
            y += 22

        cv2.imshow('gesture-calibrate', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            samples.clear()
            recognizer.reset()
        if key in (ord('p'), ord('o'), ord('f')):
            mask = recognizer._skin_mask(gesture_frame)
            contour = recognizer._largest_contour(mask)
            if contour is not None and cv2.contourArea(contour) >= recognizer.min_contour_area:
                label = {ord('p'): 'point', ord('o'): 'open_palm', ord('f'): 'fist'}[key]
                samples[label].append(recognizer.contour_to_hu(contour, gesture_frame.shape[:2]))
        if key == ord('s'):
            averaged = {label: np.mean(np.vstack(vals), axis=0) for label, vals in samples.items() if vals}
            if averaged:
                recognizer.save_templates(args.output, averaged)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
