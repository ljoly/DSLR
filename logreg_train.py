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
    currentHouseWeights = np.transpose([0, 0, 0])
    learningRate = 0.1
    iterations = 642435
    for i in range(iterations):
        currentHouseWeights = updateWeights(
            X, y, currentHouseWeights, learningRate)
    writer(houseIndex, currentHouseWeights)


def formatCurrentY(y, houseIndex):
    tmpY = list(y)
    for i in range(len(tmpY)):
        if tmpY[i] == float(houseIndex):
            tmpY[i] = 1.0
        else:
            tmpY[i] = 0.0
    return tmpY


def getMeans():
    empty_f1 = 0
    mean_f1 = 0
    empty_f2 = 0
    mean_f2 = 0
    for i, row in enumerate(rawdata):
        if row[8] == '':
            empty_f1 += 1
        else:
            mean_f1 += float(row[8])
        # if row[12] == '':
        if row[7] == '':
            empty_f2 += 1
        else:
            # mean_f2 += float(row[12])
            mean_f2 += float(row[7])            
    mean_f1 /= (len(rawdata) - empty_f1)
    mean_f2 /= (len(rawdata) - empty_f2)
    return mean_f1, mean_f2


def formatFeatures():
    # Get data[house, herbology[8], ancient runes[12]]
    # Get data[house, herbology, Defense ag.[9]]
    data = []
    # Isolating feature House
    y = []
    for _ in range(2):
        data.append([])
    mean_f1, mean_f2 = getMeans()
    for row in rawdata:
        if row[1] == 'Gryffindor':
            y.append(1.0)
        elif row[1] == 'Ravenclaw':
            y.append(2.0)
        elif row[1] == 'Slytherin':
            y.append(3.0)
        elif row[1] == 'Hufflepuff':
            y.append(4.0)
        if row[8] == '':
            data[0].append(mean_f1)
        else:
            data[0].append(float(row[8]))
        # if row[12] == '':
        if row[7] == '':        
            data[1].append(mean_f2)
        else:
            # data[1].append(float(row[12]))
            data[1].append(float(row[7]))            

    # Normalize
    for i in range(2):
        minV, maxV = ml.getMinMax(data[i])
        data[i] = ml.normalizeData(data[i], minV, maxV)

    return data, y


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Argument missing")
        exit()
    csvfile = open(sys.argv[1])
    rawdata = list(csv.reader(csvfile))
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
