import numpy as np
import cv2
import matplotlib.pyplot as plt

def chamfer_distance(img1, img2):

    resizedImg1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    resizedImg2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_AREA)

    #threshold the image
    edges1 = cv2.Canny(resizedImg1, 70, 150)
    edges2 = cv2.Canny(resizedImg2, 70, 150)

    img1Dist = cv2.distanceTransform(edges1, cv2.DIST_L2, 5)
    img2Dist = cv2.distanceTransform(edges2, cv2.DIST_L2, 5)

    chamferDistance = np.sum(np.minimum(img1Dist, img2Dist))+ np.sum(np.minimum(img2Dist, img1Dist))
    return chamferDistance


if __name__ == "__main__":
    img = cv2.imread("testImg3.jpg", 0)
    img2 = cv2.imread("testImg4.jpg", 0)
    img3 = cv2.imread("testImg5.jpg", 0)


    print(chamfer_distance(img, img2))
    print(chamfer_distance(img, img3))
