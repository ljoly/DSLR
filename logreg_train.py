import csv
import numpy as np
import ml_functions as ml

from sklearn import linear_model


csvfile = open('assets/dataset_train.csv')
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
            y.append(1.0)
        elif row[1] == 'Ravenclaw':
            y.append(2.0)
        elif row[1] == 'Slytherin':
            y.append(3.0)
        elif row[1] == 'Hufflepuff':
            y.append(4.0)
        data[1].append(float(row[8]))
        data[2].append(float(row[12]))

# Normalize
for i in range(1, 3):
    minV, maxV = ml.getMinMax(data[i])
    data[i] = ml.normalizeData(data[i], minV, maxV)

# iris = datasets.load_iris()

lr = linear_model.LogisticRegression(C=1e11)
X = np.transpose(data[1:])
y = np.transpose(y)

lr.fit(X, y)

print(lr.intercept_)

# Transpose matrix
# X = np.transpose(data[1:])

# # for i, _ in enumerate(y):
# #     y[i] = float((y[i] == 2))


# y = np.transpose(y)
# print(y)


# theta = 1.0
# learningRate = 0.1
# for i in range(300000):
#     z = np.dot(X, theta)
#     h = sigmoid(z)
#     # Gradient Descent
#     # # Partial derivative of the cost function
#     gradient = np.dot(X.T, (h - y)) / len(y)
#     print(h - y)
#     theta -= learningRate * gradient
# print(theta)
