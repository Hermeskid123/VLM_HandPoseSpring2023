import numpy as np
import cv2
import matplotlib.pyplot as plt

def directed_cd(points1, points2):   
    minDistanceForEachPoint = []
    for point in points1:
        distances = np.linalg.norm(points2 - point, axis=1)
        minDistanceForEachPoint.append(min(distances))

    directedChamferDistance = sum(minDistanceForEachPoint) / len(minDistanceForEachPoint)

    return directedChamferDistance
        
def chamfer(points1, points2):
    chamferDistance = directed_cd(points1, points2) + directed_cd(points2, points1)
    return chamferDistance

def chamfer_distance(edges1, edges2):
    nonzeroIndex1 = np.nonzero(edges1)
    nonzeroIndex2 = np.nonzero(edges2)

    nonZero1 = []
    nonZero2 = []

    for i in range(len(nonzeroIndex1[0])):
        coor = [nonzeroIndex1[0][i], nonzeroIndex1[1][i]]
        nonZero1.append(coor)
    
    for i in range(len(nonzeroIndex2[0])):
        coor = [nonzeroIndex2[0][i], nonzeroIndex2[1][i]]
        nonZero2.append(coor)

    nonZero1 = np.asarray(nonZero1)
    nonZero2 = np.asarray(nonZero2)

    return chamfer(nonZero1, nonZero2)





def chamfer_distance1(img1, img2):

    #resizedImg1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    #resizedImg2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_AREA)

    #threshold the image
    #edges1 = cv2.Canny(resizedImg1, 70, 150)
    #edges2 = cv2.Canny(resizedImg2, 70, 150)

    img1Dist = cv2.distanceTransform(img1, cv2.DIST_L2, 3)
    img2Dist = cv2.distanceTransform(img2, cv2.DIST_L2, 3)

    chamferDistance = np.sum(img1Dist * img2) + np.sum(img2Dist * img1)
    return chamferDistance


if __name__ == "__main__":
    img = cv2.imread("testImg3.jpg", 0)
    img2 = cv2.imread("testImg4.jpg", 0)
    img3 = cv2.imread("testImg5.jpg", 0)


    print(chamfer_distance(img, img2))
    print(chamfer_distance(img, img3))
