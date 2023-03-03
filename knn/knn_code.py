from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np
import utility as u
import distance_functions as df
import time

def classification(trainSet, testSet, distanceFunction):
    classifications = []
    for i in range(len(testSet[0])):
        classifications.append(nearest_neighbor(trainSet, testSet[0][i], distanceFunction))
    #print(classifications)
    classAccuracy = 0
    for i in range(len(classifications)):
        accuracy = 0
        if classifications[i] == testSet[1][i]:
            accuracy = 1
        print("Id:%5d, predicted=%s, actual=%s, accuracy=%6.4f" % (i+1, classifications[i], testSet[1][i], accuracy))
        
        classAccuracy = classAccuracy + accuracy
    print("Classification accuracy=%6.4f" % (classAccuracy))

    return classAccuracy / len(testSet[0])

def nearest_neighbor(trainingInputs, testInput, distanceFunction):
    distanceList = []
    for i in range(len(trainingInputs[0])):
        distanceList.append(distanceFunction(testInput, trainingInputs[0][i]))
        
    minDistance = np.argmin(distanceList)
    nearestNeighborClassification = trainingInputs[1][minDistance]

    return nearestNeighborClassification


def training(newTrainSet, testSet, distanceFunction, name, n, flag, mpegFlag):
    acc = []
    times = []
    for i in range(n):
        newTestSet = u.create_new_test_set(testSet, 10)

        if flag == "edge":
            newTestSet = u.edge_detection_set(newTestSet, mpegFlag)
        elif flag == "binary":
            newTestSet = u.binary_set(newTestSet)

        start  = time.time()
        acc.append(classification(newTrainSet, newTestSet, distanceFunction))
        end = time.time()
        times.append(end - start)

    acc = sum(acc) / len(acc)
    times = sum(times) / len(times)
    print("Average of accuracies for", name, "data =", acc)
    print("Average of times for", name, "data =", times)

    return (acc, times)

