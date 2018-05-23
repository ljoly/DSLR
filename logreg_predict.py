import sys
import csv
import numpy as np
import ml_functions as ml
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def showDistribution(distribution):
    kwargs = dict(histtype='stepfilled', ec='black', alpha=0.3, bins=2)
    for i in range(len(distribution)):
        plt.hist(distribution[i], **kwargs)
    plt.title('Distribution')
    plt.legend(['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff'])
    plt.xlabel('Houses', fontsize=16)
    plt.ylabel('Students', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.show()


def predict():
    f = open('assets/houses.csv', 'w')
    f.write('Index,Hogwarts House\n')
    distribution = [[], [], [], []]
    for i, studMarks in enumerate(X):
        probs = []
        for _, weights in enumerate(housesWeights):
            z = np.dot(studMarks, weights)
            probs.append(sigmoid(z))
            _, maxV = ml.getMinMax(probs)
            house = ''
            for j in range(len(probs)):
                if probs[j] == maxV:
                    house = houses[j]
        if house == 'Gryffindor':
            distribution[0].append('Gryffindor')
        elif house == 'Ravenclaw':
            distribution[1].append('Ravenclaw')
        elif house == 'Slytherin':
            distribution[2].append('Slytherin')
        elif house == 'Hufflepuff':
            distribution[3].append('Hufflepuff')
        f.write(str(i) + ',' + house + '\n')
    return distribution


def formatFeatures():
    data = []
    for _ in range(lenFeatures):
        data.append([])
    for row in rawdata:
        for i in range(lenFeatures):
            if row[i + indexFeatures] == '':
                data[i].append(0.0)
            else:
                data[i].append(float(row[i + indexFeatures]))

    # Normalize
    for i in range(lenFeatures):
        minV, maxV = ml.getMinMax(data[i])
        data[i] = ml.normalizeData(data[i], minV, maxV)

    return data


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
    data = formatFeatures()

    X = np.transpose(data)
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    csvWeights = open('assets/weights.csv')
    rawWeights = list(csv.reader(csvWeights))
    housesWeights = []

    for i in range(len(rawWeights)):
        w = []
        for j in range(1, len(rawWeights[i])):
            w.append(float(rawWeights[i][j]))
        housesWeights.append(w)

    distribution = predict()
    showDistribution(distribution)
