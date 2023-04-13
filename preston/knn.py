
from chamfer_distance import *
import numpy as np
import os
import cv2

def knn(img, datasetPath='/home/preston/Downloads/HANDPOSE_DATA/clutter_segmented', k=1):
    distances = []
    filenames = os.listdir(datasetPath)
    for filename in filenames:
        image = cv2.imread(datasetPath + "/" + filename)
        image = cv2.Canny(image, 50, 100)
        chamferDistance = chamfer_distance(img, image)
        distances.append(chamferDistance)
    
    distances = np.asarray(distances)

    kNearest = np.argsort(distances)
    kNearestNeighbors = kNearest[0:k]
    if k == 1:
        return datasetPath+"/"+filenames[int(kNearestNeighbors)]
    else:
        return kNearestNeighbors
if __name__ == "__main__":
    img = cv2.imread('/home/preston/Downloads/HANDPOSE_DATA/clutter_segmented/clutter001.bmp')
    img = cv2.Canny(img, 50, 100)
    res = knn(img)

    print(res)

