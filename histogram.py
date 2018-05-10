import csv
import sys
import ml_functions as ml
import matplotlib.pyplot as plt

csvfile = open('assets/dataset_train.csv')
rawdata = list(csv.reader(csvfile))

# Features: modify indexFeatures according to the dataset
indexFeatures = 6
lenFeatures = len(rawdata[0]) - indexFeatures
features = []
for i in range(len(rawdata[0]) - indexFeatures):
    features.append(rawdata[0][i + indexFeatures])

# Arrays of marks for each house
gryf = []
raven = []
slyth = []
huffle = []

# Array of arrays of marks
houses = [gryf, raven, slyth, huffle]
lenHouses = len(houses)

# Arrays of Std Deviations
gryfStd = []
ravenStd = []
slythStd = []
huffleStd = []

# Array of arrays of Std arrays
housesStd = [gryfStd, ravenStd, slythStd, huffleStd]

# Array of Std of Std
featuresStd = [0.0] * lenFeatures

# Array of Mean of Std
stdMean = [0.0] * lenFeatures

# Init arrays with generic size
for i in range(lenHouses):
    for _ in range(lenFeatures):
        houses[i].append([])
        housesStd[i].append([])

# main
del rawdata[0]
for row in rawdata:
    if ml.isFormatted(row):
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
        house[j] = ml.formatData(row)
        minV, maxV = ml.getMinMax(house[j])
        house[j] = ml.normalizeData(house[j], minV, maxV)
        mean = ml.getMean(house[j])
        housesStd[i][j] = ml.getStd(house[j], mean)
        stdMean[j] += housesStd[i][j]

for i, mean in enumerate(stdMean):
    stdMean[i] /= lenHouses

# Get Std
for i, house in enumerate(housesStd):
    for j, row in enumerate(house):
        featuresStd[j] += (row - stdMean[j]) * (row - stdMean[j])

for i, std in enumerate(featuresStd):
    featuresStd[i] /= lenHouses - 1
    featuresStd[i] = featuresStd[i] ** 0.5

# Get index of min std
minIndex = 0
for i in range(lenFeatures):
    if featuresStd[i] < featuresStd[minIndex]:
        minIndex = i

print('The most homogeneous feature between the four houses is:',
      features[minIndex])

# Plot
kwargs = dict(histtype='stepfilled', ec='black', alpha=0.3)

for i in range(lenFeatures):
    plt.figure(i)

    plt.hist(gryf[i], **kwargs)
    plt.hist(raven[i], **kwargs)
    plt.hist(slyth[i], **kwargs)
    plt.hist(huffle[i], **kwargs)
    plt.title(features[i])
    plt.axvline(ml.getMean(gryf[i] + raven[i] + slyth[i] +
                           huffle[i]), color='k', linestyle='dashed', linewidth=1)
    plt.legend(['Std mean', 'Gryffindor', 'Ravenclaw',
                'Slytherin', 'Hufflepuff'])
    plt.xlabel('Marks', fontsize=16)
    plt.ylabel('Students', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

plt.show()
