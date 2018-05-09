import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

csvfile = open('assets/dataset_train.csv')
rawdata = list(csv.reader(csvfile))

# Features: modify indexFeatures according to the dataset
indexFeatures = 6
lenFeatures = len(rawdata[0]) - indexFeatures
features = [] * lenFeatures
for i in range(len(rawdata[0]) - indexFeatures):
    features.append(rawdata[0][i + indexFeatures])

# Arrays of marks for each house
gryf = [[], [], [], [], [], [], [], [], [], [], [], [], []]
raven = [[], [], [], [], [], [], [], [], [], [], [], [], []]
slyth = [[], [], [], [], [], [], [], [], [], [], [], [], []]
huffle = [[], [], [], [], [], [], [], [], [], [], [], [], []]

# Array of arrays of marks
houses = [gryf, raven, slyth, huffle]
lenHouses = len(houses)

# Arrays of Std Deviations
gryfStd = [0.0] * lenFeatures
ravenStd = [0.0] * lenFeatures
slythStd = [0.0] * lenFeatures
huffleStd = [0.0] * lenFeatures

# Array of arrays of Std arrays
housesStd = [gryfStd, ravenStd, slythStd, huffleStd]

# Array of Std of Std
featuresStd = [0.0] * lenFeatures

# Array of Mean of Std
stdMean = [0.0] * lenFeatures


def normalizeData(topic, minV, maxV):
    for i, _ in enumerate(topic):
        topic[i] = (topic[i]-minV)/(maxV-minV)
    return topic


def getStd(topic, mean):
    # print(topic, mean)
    std = 0
    for mark in topic:
        std += (mark - mean) * (mark - mean)
    std /= len(topic) - 1
    std = math.sqrt(std)
    return std


def getMinMax(topic):
    # Get min and max values
    topic = sorted(topic)
    minV = topic[0]
    maxV = topic[len(topic) - 1]
    return minV, maxV


def getMean(topic):
    mean = 0
    for i in range(len(topic)):
        mean += topic[i]
    mean /= len(topic)
    return mean


def formatData(topic):
    for i, elem in enumerate(topic):
        # Replace empty strings by '0' value
        if not elem:
            topic[i] = 0.0
        else:
            topic[i] = float(elem)
    return topic

# main
del rawdata[0]
for row in rawdata:
    tmp = gryf
    if row[1] == 'Ravenclaw':
        tmp = raven
    elif row[1] == 'Slytherin':
        tmp = slyth
    elif row[1] == 'Hufflepuff':
        tmp = huffle
    for j in range(lenFeatures):
        tmp[j].append(row[j + indexFeatures])
# Get all stats
for i, house in enumerate(houses):
    for j, row in enumerate(house):
        house[j] = formatData(row)
        minV, maxV = getMinMax(house[j])
        house[j] = normalizeData(house[j], minV, maxV)
        mean = getMean(house[j])
        housesStd[i][j] = getStd(house[j], mean)
        stdMean[j] += housesStd[i][j]
        
for i, mean in enumerate(stdMean):
    stdMean[i] /= lenHouses

# Get Std
for i, house in enumerate(housesStd):
    for j, row in enumerate(house):
        featuresStd[j] += (row - stdMean[j]) * (row - stdMean[j])

for i, std in enumerate(featuresStd):
    featuresStd[i] /= lenHouses - 1
    featuresStd[i] = math.sqrt(featuresStd[i])

# Get index of min std
minIndex = 0
for i in range(lenFeatures):
    if featuresStd[i] < featuresStd[minIndex]:
        minIndex = i

kwargs = dict(histtype='stepfilled', alpha=0.3)
plt.figure()
plt.hist(gryf[minIndex], **kwargs)
plt.hist(raven[minIndex], **kwargs)
plt.hist(slyth[minIndex], **kwargs)
plt.hist(huffle[minIndex], **kwargs)
plt.title(features[minIndex])
plt.legend(['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff'])
plt.xlabel('Marks', fontsize=16)
plt.ylabel('Students', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()
