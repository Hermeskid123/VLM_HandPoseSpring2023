import numpy as np
import skimage.io as io
#import skimage.color as color
import matplotlib.pyplot as plt
from math import ceil, floor


#this funtion is used to solve a system of linear equations with 2 linear equations and 2 variables (a and b)
def solve_for_a_and_b(originalHigh, resizedHigh):
    a = np.array([[-0.5, 1], [resizedHigh, 1]])
    b = np.array([-0.5, originalHigh])
    invA = np.linalg.inv(a)
    x = invA.dot(b)
    
    return x[0], x[1]

#apply a and b to a coordinate to be able to map to the resized. We are imposing the larger image into the smaller one.
def new_coordinate(coord, a, b):
    x, y = coord
    newX = (x * a) + b
    newY = (y * a) + b
    newCoordenate = (newX, newY)
    
    return newCoordenate

#returns a list of the 4 nearest neighbors to the coordinate given, Since the coordinate might be in a decimal we simply use floor and ceiling
def four_neighbors(coord, shape):
    x, y = coord
    
    if int(floor(x)) == -1:
        x = 0
    elif int(ceil(x)) == shape[0]:
        x = shape[0] - 1
        
    if int(floor(y)) == -1:
        y = 0
    elif int(ceil(y)) == shape[1]:
        y = shape[1] - 1
        
        
    leftUp = (int(floor(x)), int(floor(y)))
    rightUp = (int(floor(x)), int(ceil(y)))
    leftDown = (int(ceil(x)), int(floor(y)))
    rightdown = (int(ceil(x)), int(ceil(y)))
    
    neighborList = []
    neighborList.append(leftUp)
    neighborList.append(rightUp)
    neighborList.append(leftDown)
    neighborList.append(rightdown)
    
    return neighborList

# This function performs bilinear interpolation it uses color values and the new coordinate
def bilinear_interpolation(colorVals, newCoord):
    topLeftC = colorVals[0]
    topRightC = colorVals[1]
    bottomRightC = colorVals[2]
    bottomLeftC = colorVals[3]
    
    h, w = newCoord
    
    weightH = h % 1
    weightW = w % 1
    
    if weightW == 0.5:
        p1 = np.floor((weightW * topLeftC) + (weightW * topRightC))
        p2 = np.floor((weightW * bottomLeftC) + (weightW * bottomRightC))
    elif weightW > 0.5:
        p1 = np.floor(((1 - weightW) * topLeftC) + (weightW * topRightC))
        p2 = np.floor(((1 - weightW) * bottomLeftC) + (weightW * bottomRightC))
    else:
        p1 = np.floor((weightW * topLeftC) + ((1 - weightW) * topRightC))
        p2 = np.floor((weightW * bottomLeftC) + ((1 - weightW) * bottomRightC))
        
    if weightH == 0.5:
        newColor = np.floor((weightH * p1) + (weightH * p2))
    elif weightH > 0.5:
        newColor = np.floor(((1 - weightH) * p1) + (weightH * p2))
    else:
        newColor = np.floor((weightH * p1) + ((1 - weightH) * p2))
        
    return newColor
            
#the resize function takes in an image as a numpy array, and a factor so a factor of 2 will double the image size.
#the factor can be any float that is greater than 0. The larger the number the longer the resize takes. This needs performance improvements
#Can take rgba images but preferable simple rgb images
#this function returns the resized image as a numpy array shape = (rows*factor, columns*factor, channels)
def resize_img(img, factor):
    h, w, c = img.shape
    hFactor = h * factor
    wFactor = w * factor
        
    resizedImg = np.zeros((int(hFactor), int(wFactor), c))
    originalHigh = h - 0.5
    resizedHigh = (h * factor) - 0.5
    a, b = solve_for_a_and_b(originalHigh, resizedHigh)
    
    rows = len(resizedImg)
    columns = len(resizedImg[0])
    
    for row in range(rows):
        for col in range(columns):
            newCoord = new_coordinate((row, col), a, b)
            fourNeighbors = four_neighbors(newCoord, (h, w))
            neighborColors = []
            
            for neighbor in fourNeighbors:
                neighborColors.append(img[neighbor[0]][neighbor[1]])
            
            newColor = bilinear_interpolation(neighborColors, newCoord)
            resizedImg[row][col] = newColor
    
    
    resizedImg = resizedImg.astype(np.uint8)
    
    #io.imshow(resizedImg)
    #plt.show()
    
    return resizedImg