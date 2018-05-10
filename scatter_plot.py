import csv
import math

csvfile = open('assets/dataset_train.csv')
rawdata = list(csv.reader(csvfile))

# Features: modify indexFeatures according to the dataset
indexFeatures = 6
lenFeatures = len(rawdata[0]) - indexFeatures
features = [] * lenFeatures
for i in range(len(rawdata[0]) - indexFeatures):
    features.append(rawdata[0][i + indexFeatures])

marks = []
stds = []

def getStd(feature, mean):
    # print(feature, mean)
    std = 0
    for mark in feature:
        std += (mark - mean) * (mark - mean)
    std /= len(feature) - 1
    std = math.sqrt(std)
    return std

def getMean(feature):
    mean = 0
    for i in range(len(feature)):
        mean += feature[i]
    mean /= len(feature)
    return mean

def normalizeData(feature, minV, maxV):
    for i, _ in enumerate(feature):
        feature[i] = (feature[i]-minV)/(maxV-minV)
    return feature

def getMinMax(feature):
    feature = sorted(feature)
    minV = feature[0]
    maxV = feature[len(feature) - 1]
    return minV, maxV


# Get data
for i in range(lenFeatures):
    marks.append([])

del rawdata[0]
for row in rawdata:
    for i in range(lenFeatures):
        # Replace empty strings by '0' value
        if not row[i + indexFeatures]:
            marks[i].append(0.0)
        else:
            # Convert to float and push to marks[]
            marks[i].append(float(row[i + indexFeatures]))

# Get stats
for i in range(lenFeatures):
    minV, maxV = getMinMax(marks[i])
    marks[i] = normalizeData(marks[i], minV, maxV)
    mean = getMean(marks[i])
    stds.append(getStd(marks[i], mean))

# Compare values
i1 = 0
i2 = 1
for i in range(lenFeatures):
    for j in range(lenFeatures):
        if stds[i] != stds[j] and abs(stds[i] - stds[j]) < abs(stds[i1] - stds[i2]):
            i1 = i
            i2 = j
print(features[i1], 'and', features[i2], 'are the same features')