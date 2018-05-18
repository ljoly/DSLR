import sys
import csv
import numpy as np
import ml_functions as ml


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def formatFeatures():
    # Get data[herbology, ancient runes]
    data = []
    for _ in range(2):
        data.append([])
    f = open('assets/unclassified_indexes.csv', 'w')
    for i, row in enumerate(rawdata):
        if row[8] != '' and row[12] != '':
            data[0].append(float(row[8]))
            data[1].append(float(row[12]))
        else:
            f.write(str(i) + '\n')

    # Normalize
    for i in range(2):
        minV, maxV = ml.getMinMax(data[i])
        data[i] = ml.normalizeData(data[i], minV, maxV)

    return data


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Argument missing")
        exit()
    csvfile = open(sys.argv[1])
    rawdata = list(csv.reader(csvfile))
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

    f = open('assets/houses.csv', 'w')
    # f.write('Index,Hogwarts House\n')
    f.write('Hogwarts House\n')    
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
        f.write(house + '\n')                
        # f.write(str(i) + ',' + house + '\n')
    
    f.close()
    csvfile = open('assets/dataset_truth2.csv')
    raw = list(csv.reader(csvfile))
    f = open('assets/dataset_truth_reformat.csv', 'w')
    for row in raw:
        f.write(row[1] + '\n')
