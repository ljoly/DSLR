import csv
import numpy as np
import ml_functions as ml

csvfile = open('assets/dataset_train2.csv')
rawdata = list(csv.reader(csvfile))
del rawdata[0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# # def costFunction(theta, X, y):

# Get data[house, herbology, ancient runes]
data = []
y = []
for i in range(3):
    data.append([])
for row in rawdata:
    if ml.isFormatted(row):
        if row[1] == 'Gryffindor':
            y.append(0.0)
        elif row[1] == 'Ravenclaw':
            y.append(1.0)
        elif row[1] == 'Slytherin':
            y.append(0.0)
        elif row[1] == 'Hufflepuff':
            y.append(0.0)
        data[1].append(float(row[8]))
        data[2].append(float(row[12]))

# Normalize
for i in range(1, 3):
    minV, maxV = ml.getMinMax(data[i])
    data[i] = ml.normalizeData(data[i], minV, maxV)

# Transpose matrix
X = np.transpose(data[1:])
# print(X)
y = np.transpose(y)

intercept = np.ones((len(X),1))
X = np.concatenate((intercept, X), axis=1)
theta = np.zeros(X.shape[1])
# print(X.shape[1])
# print(X.T)
learningRate = 0.1
while True:
    z = np.dot(X, theta)
    h = sigmoid(z)
    # Gradient descent using Partial derivative of the cost function
    gradient = np.dot(X.T, (h - y)) / len(y)
    theta -= learningRate * gradient
    if learningRate * gradient[0] <= 0.000001:
        break
print(theta)
