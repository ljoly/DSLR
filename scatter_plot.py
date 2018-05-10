import csv
import ml_functions as ml

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
    for i in range(lenFeatures):
        # Replace empty strings by '0' value
        if not row[i + indexFeatures]:
            marks[i].append(0.0)
        else:
            # Convert to float and push to marks[]
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
for i in range(lenFeatures):
    for j in range(lenFeatures):
        if stds[i] != stds[j] and abs(stds[i] - stds[j]) < abs(stds[i1] - stds[i2]):
            i1 = i
            i2 = j
print(features[i1], 'and', features[i2], 'are the same features')
