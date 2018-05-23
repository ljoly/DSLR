import sys
import csv
import numpy as np
import ml_functions as ml
from multiprocessing import Process


def writer(houseIndex, houseWeights):
    f = open('/tmp/' + houses[houseIndex] + '.csv', 'w')
    s = ''
    for i in range(len(houseWeights)):
        s += str(houseWeights[i])
        if i < len(houseWeights) - 1:
            s += ','
    s += '\n'
    f.write(s)
    f.close()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFun(X, y, weights):
    # Compute the cost of the current weights using Cross-Entropy algorithm
    m = len(X)
    z = np.dot(X, weights)
    predictions = sigmoid(z)

    # Error when y=1
    cost1 = -y*np.log(predictions)
    # Error when y=0
    cost0 = (1-y)*np.log(1-predictions)
    # Sum of both costs
    cost = cost1 - cost0
    # Average cost
    cost = np.sum(cost)/m
    return cost


def updateWeights(X, y, weights, learningRate):
    m = len(X)
    z = np.dot(X, weights)
    predictions = sigmoid(z)

    gradient = np.dot(X.T,  predictions - y) / m
    gradient *= learningRate
    weights = weights - gradient

    return weights


def train(houseIndex, data, y):
    # Transpose data into matrix
    X = np.transpose(data)
    # Transpose y into matrix for later computations
    y = np.transpose(y)

    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    # Init features' weights
    currentHouseWeights = []
    for _ in range(lenFeatures + 1):
        currentHouseWeights.append(0)
    currentHouseWeights = np.transpose(currentHouseWeights)
    learningRate = 0.1
    prevCost = 1
    maxIter = 50000
    for i in range(maxIter):
        currentHouseWeights = updateWeights(
            X, y, currentHouseWeights, learningRate)
        currentCost = costFun(X, y, currentHouseWeights)
        if abs(prevCost - currentCost) <= 0.0001:
            print(i)
            break
        prevCost = currentCost
    writer(houseIndex, currentHouseWeights)


def formatCurrentY(y, houseIndex):
    tmpY = list(y)
    for i in range(len(tmpY)):
        if tmpY[i] == float(houseIndex):
            tmpY[i] = 1.0
        else:
            tmpY[i] = 0.0
    return tmpY


def formatFeatures():
    data = []
    # Isolating feature House
    y = []
    for _ in range(lenFeatures):
        data.append([])
    for row in rawdata:
        if ml.isFormatted(row):
            if row[1] == 'Gryffindor':
                y.append(1.0)
            elif row[1] == 'Ravenclaw':
                y.append(2.0)
            elif row[1] == 'Slytherin':
                y.append(3.0)
            elif row[1] == 'Hufflepuff':
                y.append(4.0)
            for i in range(lenFeatures):
                data[i].append(float(row[i + indexFeatures]))

    # Normalize
    for i in range(lenFeatures):
        minV, maxV = ml.getMinMax(data[i])
        data[i] = ml.normalizeData(data[i], minV, maxV)

    return data, y


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Argument missing")
        exit()
    csvfile = open(sys.argv[1])
    rawdata = list(csv.reader(csvfile))
    # Features: modify indexFeatures according to the dataset
    # In that case we skip the first two features
    # The second one has a clone and the first has no impact
    indexFeatures = 8
    lenFeatures = len(rawdata[0]) - indexFeatures
    del rawdata[0]

    houses = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
    data, rawY = formatFeatures()
    for i in range(1, 5):
        y = formatCurrentY(rawY, i)
        p = Process(target=train, args=(i-1, data, y))
        p.start()
    p.join()

    # Save weights
    f = open('assets/weights.csv', 'w')
    for h in houses:
        csvfile = open('/tmp/' + h + '.csv')
        rawdata = list(csv.reader(csvfile))
        f.write(h + ',')
        for i in range(len(rawdata[0])):
            f.write(rawdata[0][i])
            if i < len(rawdata[0]) - 1:
                f.write(',')
        f.write('\n')
