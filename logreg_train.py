import csv
import numpy as np
import ml_functions as ml
from multiprocessing.dummy import Pool as ThreadPool

csvfile = open('assets/dataset_train.csv')
rawdata = list(csv.reader(csvfile))
del rawdata[0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def updateWeights(X, y, weights, learningRate):
    m = len(X)

    z = np.dot(X, weights)
    predictions = sigmoid(z)

    gradient = np.dot(X.T,  predictions - y) / m
    gradient *= learningRate
    weights = weights - gradient

    return weights


def formatFeatures(houseIndex):
    # Get data[house, herbology, ancient runes]
    data = []
    # Isolating feature House
    y = []
    for _ in range(2):
        data.append([])
    for row in rawdata:
        if ml.isFormatted(row):
            if row[1] == 'Gryffindor':
                y.append(1.0) if houseIndex == 0 else y.append(0.0)
            elif row[1] == 'Ravenclaw':
                y.append(1.0) if houseIndex == 1 else y.append(0.0)
            elif row[1] == 'Slytherin':
                y.append(1.0) if houseIndex == 2 else y.append(0.0)
            elif row[1] == 'Hufflepuff':
                y.append(1.0) if houseIndex == 3 else y.append(0.0)
            data[0].append(float(row[8]))
            data[1].append(float(row[12]))
    return data, y


def train(houseIndex):
    data, y = formatFeatures(houseIndex)
    # Normalize
    for i in range(2):
        minV, maxV = ml.getMinMax(data[i])
        data[i] = ml.normalizeData(data[i], minV, maxV)

    # Transpose data into matrix
    X = np.transpose(data)
    # Transpose y into matrix for later computations
    y = np.transpose(y)

    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    # Init features' weights
    weights = np.transpose([0, 0, 0])
    learningRate = 0.1
    iterations = 642435
    for i in range(iterations):
        weights = updateWeights(X, y, weights, learningRate)
    print('BYEBYE', houseIndex, '------>', weights)
    return weights


f = open("assets/weights.csv", "w+")

houses = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
for i in range(len(houses)):
    houseWeight = train(i)
    # Save weights
    s = houses[i]
    for j in range(len(houseWeight)):
        s += ',' + str(houseWeight[j])
    s += '\n'
    f.write(s)

# houses = [0, 1, 2, 3]
# pool = ThreadPool(len(houses))
# houseWeight = pool.map(train, houses)
# print('NON')

# pool.close()
# pool.join()

# print(houseWeight)
