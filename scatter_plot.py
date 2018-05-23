import csv
import ml_functions as ml
import matplotlib.pyplot as plt

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

# Get data
for i in range(lenFeatures):
    marks.append([])

del rawdata[0]
for row in rawdata:
    if ml.isFormatted(row):
        for i in range(lenFeatures):
            marks[i].append(float(row[i + indexFeatures]))

# Get stats
for i in range(lenFeatures):
    minV, maxV = ml.getMinMax(marks[i])
    marks[i] = ml.normalizeData(marks[i], minV, maxV)
    mean = ml.getMean(marks[i])
    stds.append(ml.getStd(marks[i], mean))

# Compare values
i1 = 0
i2 = 1
j1 = 0
j2 = 1
for i in range(lenFeatures):
    for j in range(lenFeatures):
        if stds[i] != stds[j]:
            if abs(stds[i] - stds[j]) < abs(stds[i1] - stds[i2]):
                i1 = i
                i2 = j
            elif abs(stds[i] - stds[j] > abs(stds[j1] - stds[j2])):
                j1 = i
                j2 = j
print('The most identical features are:', features[i1], 'and', features[i2])
print('The most different features are:', features[j1], 'and', features[j2])

# Plot
plt.figure(1)
plt.title('The most identical features')
plt.scatter(marks[i1], marks[i2], color=['blue', 'green'])
plt.xlabel(features[i1], fontsize=16)
plt.ylabel(features[i2], fontsize=16)

plt.figure(2)
plt.title('The most different features')
plt.scatter(marks[j1], marks[j2], color=['blue', 'green'])
plt.xlabel(features[j1], fontsize=16)
plt.ylabel(features[j2], fontsize=16)

plt.show()
