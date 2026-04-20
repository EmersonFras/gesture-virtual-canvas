"""
Live QA viewer for Task 3 gesture recognition.

Usage:
    python calibration/gesture_qa.py
    python calibration/gesture_qa.py --camera 1

Keys:
    q = quit
    r = reset temporal smoothing state
"""

import argparse
import sys

import cv2

sys.path.insert(0, '.')

from calibration.gesture import GestureRecognizer, annotate_gesture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    parser.add_argument('--templates', help='Optional JSON file with calibrated gesture templates')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open camera {args.camera}')

    recognizer = GestureRecognizer(template_path=args.templates)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        result = recognizer.update(frame)
        vis = annotate_gesture(frame, result)

        cv2.imshow('gesture-qa', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            recognizer.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
