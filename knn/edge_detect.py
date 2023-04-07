import numpy as np
import skimage.io as io
from skimage.transform import rescale
import matplotlib.pyplot as plt

"""
This function creates a Gaussian kernel
It takes in as arguments an size (int) and a standard deviation (float) that is set to 1 as default
size variable gets the value to create mgrid for x and y values for the gaussian blur
I use the np.mgrid function because it takes as values with slicing what i want every row/col to contain
an np.mgrid[0:3, 0:3] would return 2 arrays the first one would be
[
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2]
]

and the second grid would return
[
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2]
]

so when using size, lets say size = 3, size then becomes 1 because of size // 2
when I use np.mgrid[-1:2, -1:2] it creates the following grids
[
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
]

and the second grid would return
[
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
]

this is done to be able to later create a matrix with center = 0 and values increase as they get further away from the center
This gives more value to the pixel in the middle of the kernel
after making the grid we apply the the gaussian formula to x, y grids (1/(2*pi*std^2)) * e^(-(x^2 + y^2)/(2*std^2))
I use numpy to my advantage to not use loops for all x,y combinations
Once this is all calculated a gaussian kernel has been made and we return that
"""
def gaussian_kernel(size, std=1):
    size = size // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    
    denominator = 2 * np.pi * std ** 2
    temp = 1/denominator

    expNum = -(x ** 2 + y ** 2)
    expDen = 2 * std**2
    expExpression = expNum / expDen
    exponential = np.exp(expExpression)

    gaussianKernel = temp * exponential

    return gaussianKernel

"""
The convolve function only works with numpy images that are in in 2d. 
I plan on fixing to use nD images but since the nature of our project is edge detection I assume we always receive images in grayscale
This takes an img array with shape (rows, cols) notice no 3rd dimension and a kernel shape(kerRows, kerCols)
This function is pretty straight forward. We basically are apply the kernel to every inner pixel of the image (we ignore boundaries)
So this is pointwise multiplication between that section of the image and the kernel then those values are summed
That sum value is then placed in the pixel it corresponds to.
After every pixel has been evaluated i normalize the result by dividing every pixel by the max of the image
Then we apply the absolute function to every pixel to get rid of negative values 
We return the new image created by the convolution
"""
def convolve2D(img, kernel):
    img = img.astype(np.float32)
    kernel = kernel.astype(np.float32)
    rows, cols = img.shape
    kerRows, kerCols = kernel.shape
    result = np.zeros((rows, cols), dtype=np.float32)
    M = kerRows // 2
    N = kerCols // 2
    
    for i in range(M, rows - M):
        for j in range(N, cols - N):
            sliced = img[i - M:i + M + 1, j - N:j + N + 1]
            result[i, j] = np.sum(sliced * kernel)
    
    result = result / np.max(result)
    result = np.abs(result)
    
    return result

"""
Initially this function used the sobel filters but i was getting errors so instead I am using a more simple filter
This function detects the edges in the image. we detect vertical and horizontal edges imgX and imgY respectively
We apply convoultion between the original image and the kernel so that we can detect said edges
Then we apply np.hypot between the imgX and imgY this operation is basically for every point i,j in imgX we find its corresponding point i,j in imgY
then we apply sqrt(x_i,j**2 + y_i,j**2)
Then we normalize the image and multiply by 255 to get back to the grayscale [0, 255]
To get the angle theta between for every i,j pair in the matrix we apply np.arctan2() just like hypot but this time the function is
arctan(x_i,j / y_i,j)  for every point i,j in imgX and imgY NOTE: this returns in radians so we have to convert to degrees then lock angles between [0 180]
if angle theta is < 0 then we add 180 and if it is greater than 180 we simply assign 180
we return the theta matrix and the gradientImg
"""
def edge_sobel_filter(img):
    kernelX = np.array([
        [-1, 0,  1],
        [-2, 0, 2],
        [-1, 0, 1]]
    )

    kernelY = np.array([
        [1, 2,  1],
        [0, 0, 0],
        [-1, -2, -1]]
    )
    
    dx = np.array([[-1, 0, 1]])
    dy = np.array([[-1], [0], [1]])

    imgX = convolve2D(img, dx)
    imgY = convolve2D(img, dy)

    gradientImg = np.hypot(imgX, imgY)

    gradientImg = gradientImg / gradientImg.max() * 255
    theta = np.arctan2(imgX, imgY)
    
    theta = theta * 180 / np.pi
    theta[theta < 0] += 180
    theta[theta > 180] = 180

    return gradientImg, theta

"""
The non maxima supression function take as input the gradient image and the theta matrix
we create non max img with the same shape as our gradent img
and we iterate ove the whole image excluding the edges
We initialize q and r to be the max value
if the angle is 0 <= theta[i, j] < 22.5 or 157.5 <= theta [i, j] <= 180 NOTE: 22.5 corresponds to 180 / 4 since we are checking the 180 angles. and each of the elif checks that quartet angle
we set the values of r and q to the the horizontal neighbors. r the right neighbor and q the left neighbor. and so on
if the actual value of gradient norms at i,j is greater than r and q then we set the non maxima at i,j to that value else we set to 0. I believe a huge source of unoptimized behavior may come from this tho 
We return the nonMaximaSupression image
"""
def non_maxima_suppression(img, theta):
    rows, cols = img.shape
    nonMaxImg = np.zeros((rows, cols), np.int32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                q = img[i, j + 1]
                r = img[i, j - 1]
            elif (22.5 <= theta[i, j] < 67.5):
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            elif (67.5 <= theta[i, j] < 112.5):
                q = img[i + 1, j]
                r = img[i - 1, j]
            elif (112.5 <= theta[i, j] < 157.5):
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]
            
            if (img[i, j] >= q) and (img[i, j] >= r):
                nonMaxImg[i, j] = img[i, j]
            else:
                nonMaxImg[i, j] = 0

    return nonMaxImg

"""
Thresholding  takes in an nonMaxImg, a lowThresholdRatio (float), and a highThresholdRatio (float)
we find the max value of non max supression and we apply the highthrsh ratio to obtain a high threshold
same with lThreshold
We create an 0 array of shape same as the img and we set values for weak and strong
If the values from the image are >= than the highThresh then we store that pair
Same with lowThreshold if it less then hThresh and greater than lthrsh we store those values
we then update the thrshIMG at those i, j pairs and set them to the strong and weak values
return the threshold image and the strong and weak values
"""
def thresholding(img, lowThrshRatio, highThrshRatio):
    hThrsh = img.max() * highThrshRatio
    lThrsh = hThrsh * lowThrshRatio

    rows, cols = img.shape

    thrshImg = np.zeros((rows, cols), np.int32)

    weak = 20
    strong = 220

    strongI, strongJ = np.where(img >= hThrsh)
    weakI, weakJ = np.where((img <= hThrsh) & (img >= lThrsh))

    thrshImg[strongI, strongJ] = strong
    thrshImg[weakI, weakJ] = weak

    return thrshImg, strong, weak

"""
The hysteresis function we basically check whether weak edges are connected to strong edges and if they are we change their value to the strong edge
This is basically saying that they are connected. Sorry for the messy code. if that weak point is not connected to to a trong point then we get rid of it by changing it to 0
We do this for every point that is not the image corners and sides.
Once this is done then the Canny edge detection is done
"""
def hysteresis(img, weak, strong=255):
    rows, cols = img.shape
    
    for i in range(1, rows - 1):
        for j in range(1, cols -1):
            if (img[i, j] == weak):
                if ((img[i+1, j-1] == strong) 
                    or (img[i+1, j] == strong) 
                    or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) 
                    or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) 
                    or (img[i-1, j] == strong) 
                    or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    
    return img

"""
This function is the full canny edge detectiona nd it uses every function above in order
it takes in a GRAYSCALE img (it wont work with rgb) and it has optional parameters in case you want to blur the image NOTE: more blur is gotten from a bigger sized kernel
returns the edgeImg
"""
def canny_edge_detection(img, blur=False, size=5, std=1):
    if blur:
        kernel = gaussian_kernel(size, std)
        img = convolve2D(img, kernel)
    
    gradImg, theta = edge_sobel_filter(img)

    nonMaxImg = non_maxima_suppression(gradImg, theta)

    thrshImg, strong, weak = thresholding(nonMaxImg, 0.1, 0.15)

    edgeImg = hysteresis(thrshImg, weak, strong)
    
    return edgeImg

def main():
    img = io.imread("testImg2.jpg", True)
    #only rescale because the test image is otherwise too big and the program takes long to finish.
    #if you use smaller images this should not be a problem
    img = rescale(img, 0.1, anti_aliasing=False)
    
    edgeImg = canny_edge_detection(img, blur=True)

    plt.imshow(edgeImg, cmap="gray")
    plt.show()
    
if __name__ == "__main__":
    main()
