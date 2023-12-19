import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

# Loading dataset
iris = datasets.load_iris()
X, y = iris.data, (iris.target == 2).astype(int)

# Split the dataset into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Cost(X_train, Y_train, m): 
    cost_ = 0
    N = X_train.shape[0] 
    for i in range(N):
        agg = (X_train[i] * m).sum()
        h = Sigmoid(agg)
        cost = -Y_train[i] * np.log(h) - (1 - Y_train[i]) * np.log(1 - h) 
        cost_ += cost
    return cost_

def Step_Gradient(X_train, Y_train, lr, m): 
    N = X_train.shape[0]
    slope_m = np.zeros(X_train.shape[1])
    for i in range(N):
        agg = (X_train[i] * m).sum()
        h = Sigmoid(agg)
        slope_m += (-1 / N) * (Y_train[i] - h) * X_train[i]
    m = m - lr * slope_m 
    return m

def Fit(X_train, Y_train, epochs=100, lr=0.01): 
    m = np.zeros(X_train.shape[1])
    cost_array = []
    unit = epochs // 100
    for i in range(epochs):
        m = Step_Gradient(X_train, Y_train, lr, m) 
        cost_ = Cost(X_train, Y_train, m) 
        cost_array.append(cost_)
        if i % unit == 0:
            print("Epoch:{}, Cost:{}".format(i, cost_)) 
    return m, cost_array

def Predict(X_test, m): 
    y_pred = []
    N = X_test.shape[0] 
    for i in range(N):
        agg = (X_test[i] * m).sum() 
        h = Sigmoid(agg)
        if h >= 0.5:
            y_pred.append(1) 
        else:
            y_pred.append(0) 
    return np.array(y_pred)

def Accuracy(Y_test, Y_pred): 
    correct = 0
    N = Y_test.shape[0]
    correct = (Y_test == Y_pred).sum()
    return (correct / N) * 100

m, cost_array = Fit(X_train, Y_train, 5000, 0.01) 
print(m)
plt.plot(cost_array)
plt.grid()
plt.show()
Y_pred_train = Predict(X_train, m) 
Accuracy(Y_train, Y_pred_train) 
Y_pred_val = Predict(X_test, m)