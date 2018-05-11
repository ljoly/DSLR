import csv
import pandas as pd
import seaborn as sns
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
features.append(rawdata[0][1])
del rawdata[0]

marks = []

# Get data
for i in range(lenFeatures + 1):
    marks.append([])

for row in rawdata:
    if ml.isFormatted(row):
        marks[lenFeatures].append(row[1])
        for i in range(lenFeatures):
            marks[i].append(float(row[i + indexFeatures]))

# Normalize
for i in range(lenFeatures):
    minV, maxV = ml.getMinMax(marks[i])
    marks[i] = ml.normalizeData(marks[i], minV, maxV)

# Prepare data for pair plot
pairplot = []
for i in range(len(marks[0])):
    tmp = []
    for j in range(lenFeatures + 1):
        tmp.append(marks[j][i])
    pairplot.append(tmp)

# Plot
pairplot = pd.DataFrame(pairplot, columns=features)
sns_plot = sns.pairplot(pairplot, size=2.5, hue=features[lenFeatures])
sns_plot.savefig("pair_plot.png")
