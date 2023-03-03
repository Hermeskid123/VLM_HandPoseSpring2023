import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("testImg2.jpg", 0)

factor = 0.1
width = int(img.shape[1]*factor)
height = int(img.shape[0]*factor)
dim = (width, height)

resizedImg = cv.resize(img, dim, interpolation=cv.INTER_AREA)

edges = cv.Canny(resizedImg, 50, 100)

plt.imshow(edges, cmap="gray")
#plt.subplot(122), plt.imshow(edges, cmap="gray")

plt.show()