import csv
import sys
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
gryf = [[] * lenFeatures]
raven = [[] * lenFeatures]
slyth = [[] * lenFeatures]
huffle = [[] * lenFeatures]

# Array of houses
houses = [gryf, raven, slyth, huffle]


def normalizeData(topic, minV, maxV):
    for i, _ in enumerate(topic):
        topic[i] = (topic[i]-minV)/(maxV-minV)
    return topic


def prepareData(topic):
    for i, elem in enumerate(topic):
        # Replace empty strings by '0' value
        if not elem:
            topic[i] = 0.0
        else:
            topic[i] = float(elem)
    # Get min and max values
    topic = sorted(topic)
    minV = topic[0]
    maxV = topic[len(topic) - 1]
    return minV, maxV


del rawdata[0]
for row in rawdata:
    tmp = [[] * lenFeatures]
    if row[1] == 'Gryffindor':
        tmp = gryf
    elif row[1] == 'Ravenclaw':
        tmp = raven
    elif row[1] == 'Slytherin':
        tmp = slyth
    elif row[1] == 'Hufflepuff':
        tmp = huffle
    for j, _ in enumerate(tmp):
        tmp[j].append(row[j + indexFeatures])

for house in houses:
    for row in house:
        minV, maxV = prepareData(row)
        row = normalizeData(row, minV, maxV)

# plt.title('Houses')
plt.hist(gryf)
plt.hist(raven)
plt.hist(slyth)
plt.hist(huffle)
plt.legend(['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff'])
plt.xlabel('Marks', fontsize=16)
plt.ylabel('Students', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
