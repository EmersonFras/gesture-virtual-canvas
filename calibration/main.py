"""
Gesture canvas main entry point.

Usage:
    python calibration/main.py                   # webcam
    python calibration/main.py --video path/to/vid
    python calibration/main.py --camera 1

Keys: q=quit, s=save frame, c=confirm calibration step
"""

import argparse, sys, os
import cv2, numpy as np

sys.path.insert(0, '.')
from meanshift import circularNeighbors, colorHistogram, meanShiftWeights
from detector import detect_canvas
from homography import apply_homography
from gesture import GestureRecognizer, annotate_gesture

MAX_DIM = 1280
QA_DIR = 'qa_output'
COLORS = {'TL': (0,0,255), 'TR': (0,255,0), 'BR': (255,0,0), 'BL': (0,255,255)}

# Calibration phases
PHASE_WORKSPACE = 0
PHASE_BRUSH     = 1
PHASE_DRAWING   = 2


def fit(frame):
    h, w = frame.shape[:2]
    s = min(MAX_DIM / max(h, w), 1.0)
    return cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1 else frame

def draw(canvas, x, y, prevx, prevy, color):
    cv2.line(canvas, (int(prevx), int(prevy)), (int(x), int(y)), color, thickness=10)
    return canvas

def find_brush(x, y, q_model, bg):
    for _ in range(1):
        bgFeatures = circularNeighbors(bg, x, y, 15)
        p_test = colorHistogram(bgFeatures, 16, x, y, 15)
        weights = meanShiftWeights(bgFeatures, q_model, p_test, 16)
        weights = weights.flatten()
        new_x = np.sum(bgFeatures[:,0] * weights) / (np.sum(weights) + 0.0001)
        new_y = np.sum(bgFeatures[:,1] * weights) / (np.sum(weights) + 0.0001)
        if new_x + new_y > 1:
            x = new_x
            y = new_y
    return x, y

def overlay_text(vis, lines, start_y=30, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        y = start_y + i * 28
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


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
        vis = frame.copy()
        cv2.imshow('detection', vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            os.makedirs(QA_DIR, exist_ok=True)
            cv2.imwrite(f'{QA_DIR}/qa_0.png', vis)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    save_idx = 0

    phase = PHASE_WORKSPACE
    brushx = 0
    brushy = 0
    prevx  = 0
    prevy  = 0
    canvas = None
    brushHist = None

    canvas_result     = None
    detect_counter    = 0
    DETECT_INTERVAL   = 20
    canvas_just_reset = False

    recognizer = GestureRecognizer(template_path="calibration/gesture_templates.json")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = fit(frame)
        vis = frame.copy()

        # --- Phase 0: workspace detection ---
        if phase == PHASE_WORKSPACE:
            detect_counter += 1
            if detect_counter >= DETECT_INTERVAL:
                detect_counter = 0
                result = detect_canvas(frame)
                if result is not None:
                    prev_w = canvas_result["canvas_w"] if canvas_result else None
                    prev_h = canvas_result["canvas_h"] if canvas_result else None
                    canvas_result = result
                    if prev_w != canvas_result["canvas_w"] or prev_h != canvas_result["canvas_h"]:
                        canvas = np.zeros(
                            (canvas_result["canvas_h"], canvas_result["canvas_w"], 3),
                            dtype=np.uint8,
                        )
                        canvas_just_reset = True

            if canvas_result is not None:
                corner_labels = ['TL', 'TR', 'BR', 'BL']
                for label, pt in zip(corner_labels, canvas_result["corners_img"]):
                    cx_pt, cy_pt = int(pt[0]), int(pt[1])
                    cv2.circle(vis, (cx_pt, cy_pt), 8, COLORS[label], -1)
                    cv2.putText(vis, label, (cx_pt + 10, cy_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[label], 2)
                overlay_text(vis, ["Workspace detected - press 'c' to confirm"], color=(0, 255, 0))
            else:
                overlay_text(vis, ["Searching for workspace..."], color=(0, 200, 255))

        # --- Phase 1: brush calibration ---
        elif phase == PHASE_BRUSH:
            if canvas_result is not None:
                corner_labels = ['TL', 'TR', 'BR', 'BL']
                for label, pt in zip(corner_labels, canvas_result["corners_img"]):
                    cx_pt, cy_pt = int(pt[0]), int(pt[1])
                    cv2.circle(vis, (cx_pt, cy_pt), 8, COLORS[label], -1)
            cv2.circle(vis, (350, 250), 30, (0, 0, 255), 5)
            overlay_text(vis, ["Hold brush tip in circle - press 'c' to calibrate"], color=(0, 200, 255))

        # --- Phase 2: active drawing ---
        elif phase == PHASE_DRAWING:
            half_w = frame.shape[1] // 2
            cv2.line(vis, (half_w, 0), (half_w, vis.shape[0]), (100, 100, 100), 1)
            gesture = recognizer.update(frame[:, :half_w])
            vis = annotate_gesture(vis, gesture)

            if canvas is None:
                canvas = np.zeros_like(frame)

            if gesture.command == "clear":
                canvas[:] = 0

            brushx, brushy = find_brush(brushx, brushy, brushHist, frame)

            if canvas_result is not None:
                draw_x,      draw_y      = apply_homography(canvas_result["H"], (brushx, brushy))
                prev_draw_x, prev_draw_y = apply_homography(canvas_result["H"], (prevx,  prevy))
            else:
                draw_x,      draw_y      = brushx, brushy
                prev_draw_x, prev_draw_y = prevx,  prevy

            if not canvas_just_reset:
                if gesture.command == "draw":
                    canvas = draw(canvas, draw_x, draw_y, prev_draw_x, prev_draw_y, (255, 0, 0))
                elif gesture.command == "erase":
                    canvas = draw(canvas, draw_x, draw_y, prev_draw_x, prev_draw_y, (0, 0, 0))
            canvas_just_reset = False

            prevx, prevy = brushx, brushy

            if canvas_result is not None:
                warped = cv2.warpPerspective(
                    canvas,
                    canvas_result["H_inv"],
                    (vis.shape[1], vis.shape[0]),
                )
                mask = warped.any(axis=2)
                vis[mask] = warped[mask]
            else:
                canvas_drawing = np.where(canvas > 0)
                vis[canvas_drawing] = canvas[canvas_drawing]

        if canvas is None:
            canvas = np.zeros_like(frame)

        cv2.imshow('Main', vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('s'):
            os.makedirs(QA_DIR, exist_ok=True)
            cv2.imwrite(f'{QA_DIR}/qa_{save_idx}.png', vis)
            save_idx += 1

        if key == ord('c'):
            if phase == PHASE_WORKSPACE and canvas_result is not None:
                phase = PHASE_BRUSH
            elif phase == PHASE_BRUSH:
                brushFeatures = circularNeighbors(frame, 350, 250, 15)
                brushHist = colorHistogram(brushFeatures, 16, 350, 250, 15)
                brushx = 350
                brushy = 250
                prevx = 350
                prevy = 250
                phase = PHASE_DRAWING

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
