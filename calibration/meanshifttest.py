import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
sys.path.insert(0,'.')
from meanshift import circularNeighbors, meanShiftWeights, colorHistogram
radius = 10


bg = cv2.imread('test-images/backgroundwtarget1.png')
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

brush = cv2.imread('test-images/target.png')
brush = cv2.cvtColor(brush, cv2.COLOR_BGR2RGB)

brushFeatures = circularNeighbors(brush,30,30,radius)
q_model = colorHistogram(brushFeatures,16,30,30,radius)
currentX = 460
currentY = 240

for i in range(25):
    bgFeatures = circularNeighbors(bg, currentX, currentY, radius)
    p_test = colorHistogram(bgFeatures,16,currentX,currentY,radius)
    weights = meanShiftWeights(bgFeatures,q_model,p_test,16)
    weights = weights.flatten()
    currentX = np.sum(bgFeatures[:,0] * weights) / (np.sum(weights) + 0.0001)
    currentY = np.sum(bgFeatures[:,1] * weights) / (np.sum(weights) + 0.0001)
fig, (ax1,ax2) = plt.subplots(1,2)
print('x',currentX,'y',currentY)
ax2.imshow(brush.astype('uint8'))
ax1.imshow(bg.astype('uint8'))
ax1.plot(currentX,currentY,'b+',markersize =20, linewidth=2)
ax1.set_aspect('equal')
plt.show()



