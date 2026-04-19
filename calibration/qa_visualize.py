"""
QA viewer for Task 2 canvas detection.

Usage:
    python calibration/qa_visualize.py                        # webcam
    python calibration/qa_visualize.py --image path/to/img   # static image
    python calibration/qa_visualize.py --video path/to/vid   # video file

Keys: q=quit, s=save frame
"""

import argparse, sys, os
import cv2, numpy as np

sys.path.insert(0, '.')
from calibration.detector import detect_canvas

MAX_DIM = 1280
QA_DIR = 'qa_output'
COLORS = {'TL': (0,0,255), 'TR': (0,255,0), 'BR': (255,0,0), 'BL': (0,255,255)}


def fit(frame):
    h, w = frame.shape[:2]
    s = min(MAX_DIM / max(h, w), 1.0)
    return cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1 else frame


def annotate(frame):
    result = detect_canvas(frame)
    vis = frame.copy()
    if result is None:
        cv2.putText(vis, 'NO DETECTION', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return vis, None

    corners = result['corners_img']
    pts = corners.reshape((-1,1,2)).astype(np.int32)
    cv2.polylines(vis, [pts], True, (0,255,255), 2)
    for i, (label, color) in enumerate(COLORS.items()):
        x, y = int(corners[i][0]), int(corners[i][1])
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.putText(vis, label, (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, 'DETECTED', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    rect = cv2.warpPerspective(frame, result['H'], (result['canvas_w'], result['canvas_h']))
    return vis, rect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image')
    ap.add_argument('--video')
    ap.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f'Cannot read {args.image}')
        frame = fit(frame)
        vis, rect = annotate(frame)
        cv2.imshow('detection', vis)
        if rect is not None:
            cv2.imshow('rectified', rect)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            os.makedirs(QA_DIR, exist_ok=True)
            cv2.imwrite(f'{QA_DIR}/qa_0.png', vis)
            if rect is not None:
                cv2.imwrite(f'{QA_DIR}/qa_rect_0.png', rect)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    save_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = fit(frame)
        vis, rect = annotate(frame)
        cv2.imshow('detection', vis)
        if rect is not None:
            cv2.imshow('rectified', rect)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            os.makedirs(QA_DIR, exist_ok=True)
            cv2.imwrite(f'{QA_DIR}/qa_{save_idx}.png', vis)
            if rect is not None:
                cv2.imwrite(f'{QA_DIR}/qa_rect_{save_idx}.png', rect)
            save_idx += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
