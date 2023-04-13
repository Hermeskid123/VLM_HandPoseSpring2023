from chamfer_distance import *
import numpy as np
import os
import cv2

def knn(img, datasetPath, k):
    distances = []
    filenames = os.listdir(datasetPath)

    for filename in filenames:
        image = cv2.imread(datasetPath + "/" + filename)
        chamferDistance = chamfer_distance(img, image)
        distances.append(chamferDistance)
    
    distances = np.asarray(distances)

    kNearest = np.argsort(distances)

    kNearestNeighbors = kNearest[0:k]

    return kNearestNeighbors

