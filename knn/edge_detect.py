import numpy as np
import skimage.io as io
from skimage.transform import rescale
import skimage.filters
import matplotlib.pyplot as plt
from image_resize import resize_img
from scipy import ndimage

def kernel(size, std=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]

    denominator = 2 * np.pi * std ** 2
    temp = 1/denominator

    expNum = -(x ** 2 + y ** 2)
    expDen = 2 * std**2
    expExpression = expNum / expDen

    exponential = np.exp(expExpression)

    gaussianKernel = temp * exponential

    return gaussianKernel

def gaussian_blur(img):
    blurredImg = skimage.filters.gaussian(img, sigma=(3.0, 3.0), truncate=3.5, channel_axis=2)
    return blurredImg

def edge_sobel_filter(img):
    kernelX = np.array([
        [-1, 0,  1],
        [-2, 0, 2],
        [-1, 0, 1]], np.float32
    )

    kernelY = np.array([
        [1, 2,  1],
        [0, 0, 0],
        [-1, -2, -1]], np.float32
    )

    imgX = ndimage.convolve(img, kernelX)
    imgY = ndimage.convolve(img, kernelY)

    gradientImg = np.hypot(imgX, imgY)

    gradientImg = gradientImg / gradientImg.max() * 255
    theta = np.arctan2(imgX, imgY)

    return gradientImg, theta

def non_maxima_suppression(img, theta):
    rows, cols = img.shape
    nonMaxImg = np.zeros((rows, cols), np.int32)
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j + 1]
                r = img[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = img[i + 1, j]
                r = img[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]
            
            if (img[i, j] >= q) and (img[i, j] >= r):
                nonMaxImg[i, j] = img[i, j]
            else:
                nonMaxImg[i, j] = 0
    
    plt.imshow(nonMaxImg, cmap="gray")
    plt.show()

    return nonMaxImg

def thresholding(img, lowThrshRatio, highThrshRatio):
    hThrsh = img.max() * highThrshRatio
    lThrsh = hThrsh * lowThrshRatio

    print(hThrsh, lThrsh)

    rows, cols = img.shape

    thrshImg = np.zeros((rows, cols), np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strongI, strongJ = np.where(img >= hThrsh)
    zerosI, zerosJ = np.where(img < lThrsh)
    weakI, weakJ = np.where((img <= hThrsh) & (img >= lThrsh))

    thrshImg[strongI, strongJ] = strong
    thrshImg[weakI, weakJ] = weak

    plt.imshow(thrshImg, cmap="gray")
    plt.show()

    return thrshImg, strong, weak
    
def hysteresis(img, weak, strong=255):
    rows, cols = img.shape
    print(weak, strong)
    
    for i in range(1, rows - 1):
        for j in range(1, cols -1):
            if (img[i, j] == weak):
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    
    return img

def main():
    img = io.imread("testImg2.jpg", True)
    img = rescale(img, 0.1, anti_aliasing=True)

    img = gaussian_blur(img)

    img, theta = edge_sobel_filter(img)

    img = non_maxima_suppression(img, theta)

    img, strong, weak = thresholding(img, 0.15, 0.2)

    img = hysteresis(img, weak, strong)

    plt.imshow(img, cmap="gray")
    plt.show()
    

if __name__ == "__main__":
    main()
