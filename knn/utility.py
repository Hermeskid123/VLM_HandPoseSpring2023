import numpy as np
import csv
from matplotlib import pyplot
from PIL import Image
import os
import distance_functions as df

#creates a training and testing set of 100 points each
#training set has 10 images per class
#testing set a random sampling of 100 values from the original dataset
def create_new_train_set(trainSet, n):
    (trainX, trainLabel) = trainSet
    newTrainInput = []
    newTrainingLabel = []

    for i in range(n):
        totalList = np.where(trainLabel == i)
        numbers = np.random.randint(len(totalList[0]), size=n)

        for j in range(n):
            newTrainInput.append(trainX[totalList[0][numbers[j]]])
            newTrainingLabel.append(i)

    newTrainSet = [np.array(newTrainInput), np.array(newTrainingLabel)]
    return newTrainSet

def create_new_test_set(testSet, n):
    (testX, testLabel) = testSet
    newTestInput = []
    newTestLabel = []

    randNumbers = np.random.randint(len(testX), size=n*n)
    for j in randNumbers:
        newTestInput.append(testX[j])
        newTestLabel.append(testLabel[j])
    newTestSet = [np.array(newTestInput), np.array(newTestLabel)]
    
    return newTestSet

#takes an array and returns the list of points that have a higer value than 0
def extract_point(inputVal):
    points = []
    pointList = np.where(inputVal > 0)
    for i in range(len(pointList[0])):
        points.append((pointList[0][i], pointList[1][i]))
    
    return points

def read_csv(file):
    points = []
    with open(file) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            points.append((int(row[0]), int(row[1])))
    print(points)

    return points

def binarize(image):
    threshold = 128
    binaryImage = (image > threshold) * 255
    return binaryImage

def edge_detection(dataPoint, mpegFlag):
    if mpegFlag == False:
        binaryImage = binarize(dataPoint)
    else:
        binaryImage = dataPoint
    edgeImage = binaryImage.copy()

    for i in range(len(binaryImage)):
        for j in range(len(binaryImage[0])):
            if binaryImage[i][j] > 0:
                #print(binaryImage[i][j])
                #checks if the pixel is the first column. If it is we cant check left pixel and special cases at
                #j == 0 and j == len(binaryImage) - 1 (top left pixel, bottom left pixel)
                if i == 0:
                    #first pixel so we check right and down
                    if j == 0:
                        if binaryImage[i + 1][j] > 0:
                            if binaryImage[i][j + 1] > 0:
                                edgeImage[i][j] = 0
                    #bottom left pixel so we check for right and up
                    elif j == (len(binaryImage[0]) - 1):
                        if binaryImage[i + 1][j] > 0:
                            if binaryImage[i][j - 1] > 0:
                                edgeImage[i][j] = 0
                    #this is now the first column so we check right, up, and down
                    else:
                        if binaryImage[i + 1][j] > 0:
                            if binaryImage[i][j - 1] > 0:
                                if binaryImage[i][j + 1] > 0:
                                    edgeImage[i][j] = 0
                #This is for the last column of the image with similar test cases at j == 0 and j == len(binaryImage) - 1
                #top right corner and bottom right corner
                if i == (len(binaryImage) - 1):
                    if j == 0:
                        if binaryImage[i - 1][j] > 0:
                            if binaryImage[i][j + 1] > 0:
                                edgeImage[i][j] = 0
                    elif j == (len(binaryImage[0]) - 1):
                        if binaryImage[i - 1][j] > 0:
                            if binaryImage[i][j - 1] > 0:
                                edgeImage[i][j] = 0
                    else:
                        if binaryImage[i - 1][j] > 0:
                            if binaryImage[i][j - 1] > 0:
                                if binaryImage[i][j + 1] > 0:
                                    edgeImage[i][j] = 0
                if j == 0:
                    #first pixel so we check right and down
                    if i == 0:
                        pass
                    #bottom left pixel so we check for right and up
                    elif i == (len(binaryImage) - 1):
                        pass
                    #this is now the first column so we check right, up, and down
                    else:
                        if binaryImage[i - 1][j] > 0:
                            if binaryImage[i + 1][j] > 0:
                                if binaryImage[i][j + 1] > 0:
                                    edgeImage[i][j] = 0
                
                if j == (len(binaryImage[0]) - 1):
                    #first pixel so we check right and down
                    if i == 0:
                        pass
                    #bottom left pixel so we check for right and up
                    elif i == (len(binaryImage) - 1):
                        pass
                    #this is now the first column so we check right, up, and down
                    else:
                        if binaryImage[i - 1][j] > 0:
                            if binaryImage[i + 1][j] > 0:
                                if binaryImage[i][j - 1] > 0:
                                    edgeImage[i][j] = 0

                #non image edge    
                if j != 0 and i != 0:
                    if j != len(binaryImage[0]) - 1 and i != len(binaryImage) - 1:
                        #print((binaryImage[len(binaryImage) - 1][j + 1]))
                        #print(j + 1)
                        if binaryImage[i - 1][j] > 0:
                            if binaryImage[i][j - 1] > 0:
                                if binaryImage[i + 1][j] > 0:
                                    if binaryImage[i][j + 1] > 0:
                                        edgeImage[i][j] = 0


    return edgeImage

def binary_set(dataSet):
    binarySet = []
    for i in range(len(dataSet[0])):
        binarySet.append(binarize(dataSet[0][i]))

    binarySet = np.array(binarySet)
    newSet = [binarySet, dataSet[1]]
    return newSet

def edge_detection_set(dataSet, mpegFlag):
    edgeDetectedSet = []
    for i in range(len(dataSet[0])):
        edgeDetectedSet.append(edge_detection(dataSet[0][i], mpegFlag))
    
    edgeDetectedSet = np.array(edgeDetectedSet)

    newSet = [edgeDetectedSet, dataSet[1]]
    return newSet

def grassfire_transform(image, mnistFlag):
    transform = image.copy()
    if mnistFlag == True:
        transform = np.kron(transform, np.ones((5, 5)))
    
    for i in range(0, len(transform[0])):
        for j in range(0, len(transform)):
            if transform[j][i] > 0:
                if i == 0:
                    if j == 0:
                        transform[j][i] = 1
                    else:
                        transform[j][i] = 1 + min(0, transform[j - 1][i])
                if j == 0:
                    if i == 0:
                        pass
                    else:
                        transform[j][i] = 1 + min(transform[j][i - 1], 0)
                if i != 0 and j != 0:
                    transform[j][i] = 1 + min(transform[j][i - 1], transform[j - 1][i])
    for i in range(len(transform[0]) - 1, -1, -1):
        for j in range(len(transform) - 1, -1, -1):
            if transform[j][i] > 0:
                transform[j][i] = min(transform[j][i], 1 + min(transform[j + 1][i], transform[j][i + 1]))
    
    #transform = (transform > 6) * 255
    return transform


def load_mpeg():
    files = os.listdir("mpeg7/original")

    print(len(files))

    dataset = []
    classification = []
    files.sort()
    for file in files:
        path = "mpeg7/original/" + file
        image = Image.open(path)
        classif = file.split("-")
        data = np.array(image)
        dataset.append(data)
        classification.append(classif[0])

    #dataset = np.array(dataset)

    newSet = [dataset, classification]

    return newSet     

def show_image(image):
    pyplot.imshow(image)
    pyplot.show()