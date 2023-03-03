import numpy as np

def euclidean_distance(inputX, inputY):
    distance = 0
    #print(len(inputX))
    #print(len(inputX[0]))
    for j in range(len(inputX)):
        for i in range(len(inputX[0])):
            x = int(inputX[j][i]) - int(inputY[j][i])
            x = x**2
            distance = distance + x
    distance = np.sqrt(distance)
    return distance

def manhattan_distance(inputX, inputY):
    distance = 0 
    for j in range(len(inputX)):
        for i in range(len(inputX[0])):
            x = int(inputX[j][i]) - int(inputY[j][i])
            x = np.abs(x)
            distance = distance + x
    return distance  

def measure_distance(point1, point2):
    first = (int(point1[0]) - int(point2[0])) ** 2
    second = (int(point1[1]) - int(point2[1])) ** 2
    return np.sqrt(first + second) 

def directed_chamfer(inputX, inputY):
    minDistanceForEachPoint = []
    for point in inputX:
        distances = []
        for point2 in inputY:
            distances.append(measure_distance(point, point2))
        minDistanceForEachPoint.append(min(distances))

    directedChamfer = sum(minDistanceForEachPoint) / len(minDistanceForEachPoint) 

    return directedChamfer    

def chamfer_distance(inputX, inputY):
    chamferDistance = directed_chamfer(inputX, inputY) + directed_chamfer(inputY, inputX)
    return chamferDistance

def chamfer(inputX, inputY):
    pointsX = []
    pointsY = []
    for j in range(len(inputX)):
        for i in range(len(inputX[0])):
            pointX = inputX[j][i]
            pointY = inputY[j][i]

            if pointX > 75:
                pointsX.append((j, i))
            if pointY > 75:
                pointsY.append((j, i))
    
    distance = chamfer_distance(pointsX, pointsY)

    return distance
    