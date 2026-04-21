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
from meanshift import circularNeighbors, colorHistogram, meanShiftWeights
from detector import detect_canvas

MAX_DIM = 1280
QA_DIR = 'qa_output'
COLORS = {'TL': (0,0,255), 'TR': (0,255,0), 'BR': (255,0,0), 'BL': (0,255,255)}


def fit(frame):
    h, w = frame.shape[:2]
    s = min(MAX_DIM / max(h, w), 1.0)
    return cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1 else frame

def draw(canvas,x,y,prevx,prevy,color):
    cv2.line(canvas, (int(prevx),int(prevy)), (int(x),int(y)),color,thickness = 10)

    return canvas

def find_brush(x,y,q_model,bg):
    for i in range(1):


        #basic MST
        bgFeatures = circularNeighbors(bg, x, y, 15)
        p_test = colorHistogram(bgFeatures,16,x,y,15)
        weights = meanShiftWeights(bgFeatures,q_model,p_test,16)
        weights = weights.flatten()
        new_x = np.sum(bgFeatures[:,0] * weights) / (np.sum(weights) + 0.0001)
        new_y = np.sum(bgFeatures[:,1] * weights) / (np.sum(weights) + 0.0001)

        #doesnt update if brush isnt found. If it isnt found, it defaults to 0, so if it is more than 0 it updates the brush position
        if new_x + new_y > 1:
            x = new_x
            y = new_y
    return x,y


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


    #variable declaration
    calibrated = 0
    brushx = 0
    brushy = 0
    prevx = 0
    prevy = 0
    canvas = None

    #this is the brush color, change this to (0,0,0) to erase
    color = (255,100,100)
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        frame = fit(frame)
        vis = frame.copy()
        
        #canvas just stores drawing, should be 0 where empty
        if canvas is None:
            canvas = np.zeros_like(frame)

        #if we havent calibrated the brush, draw the target circle.
        if calibrated == 0:
            cv2.circle(vis, (350,250),30,(0,0,255),5)

        if calibrated == 1:

            #Updates brush position
            brushx,brushy = find_brush(brushx,brushy,brushHist,frame)
        
            #updates the canvas with the newest point
            canvas = draw(canvas,brushx,brushy,prevx,prevy,color)

            #stores previous coordinates so a line could be drawn between them
            prevx,prevy = brushx,brushy

            #add canvas to visual frame
            canvas_drawing = np.where(canvas>0)
            vis[canvas_drawing] = canvas[canvas_drawing]
            
            

        cv2.imshow('Main', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            os.makedirs(QA_DIR, exist_ok=True)
            cv2.imwrite(f'{QA_DIR}/qa_{save_idx}.png', vis)
            save_idx += 1

        #this is to calibrate, to calibrate, hold the tip of the brush so it fills the red circle and press c.
        if key == ord('c') and calibrated == 0:
            calibrated = 1

            #creates q_model out of the area in frame
            brushFeatures = circularNeighbors(frame,350,250,15)
            brushHist = colorHistogram(brushFeatures,16,350,250,15)

            #creates initial brush conditions
            brushx = 350
            prevx = 350
            brushy = 250
            prevy = 250
        
        

            
            
        
        

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
