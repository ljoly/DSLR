import sys
import csv
import numpy as np
import ml_functions as ml


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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
    print(empty_f1, empty_f2)
    mean_f1 /= (len(rawdata) - empty_f1)
    mean_f2 /= (len(rawdata) - empty_f2)
    return mean_f1, mean_f2


def formatFeatures():
    # Get data[house, herbology[8], ancient runes[12]]
    # Get data[house, herbology, Defense ag.[9]]
    data = []
    for _ in range(2):
        data.append([])
    # f = open('assets/unclassified_indexes.csv', 'w')
    mean_f1, mean_f2 = getMeans()
    for i, row in enumerate(rawdata):
        if row[8] == '':
            data[0].append(mean_f1)
        else:
            data[0].append(float(row[8]))
        # if row[12] == '':
        if row[7] == '':        
            data[1].append(mean_f2)
        else:
            data[1].append(float(row[7]))            
            # data[1].append(float(row[12]))


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
    f.write('Index,Hogwarts House\n')
    # f.write('Hogwarts House\n')    
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
        # f.write(house + '\n')                
        f.write(str(i) + ',' + house + '\n')
    
    f.close()
    csvfile = open('assets/dataset_truth2.csv')
    raw = list(csv.reader(csvfile))
    f = open('assets/dataset_truth_reformat.csv', 'w')
    for row in raw:
        f.write(row[1] + '\n')
