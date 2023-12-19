#Simple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simple_linear_regression(X, y): 
    # Calculate the mean of X and y 
    mean_X = np.mean(X)
    mean_y = np.mean(y)
    # Calculate the slope (m) and y-intercept (b) using the least squares method 
    numerator = np.sum((X - mean_X) * (y - mean_y))
    denominator = np.sum((X - mean_X) ** 2)
    slope = numerator / denominator 
    intercept = mean_y - slope * mean_X
    return slope, intercept

def plot_regression_line(X, y, slope, intercept):
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, slope * X + intercept, color='red', label='Regression Line') 
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Generate some random data
np.random.seed(42)
X = np.random.rand(100, 1) * 10 

# Random values for X
y = 3 * X + 2 + np.random.randn(100, 1) * 2 

# Linear relationship with noise 
slope, intercept = simple_linear_regression(X, y)
print("Slope:", slope) 
print("Intercept:", intercept)
plot_regression_line(X, y, slope, intercept)


#Multiple Linear Regression
# Importing libraries
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print()

# Loading dataset
diabetes = datasets.load_diabetes()

# print(diabetes.DESCR)
X = diabetes.data 
Y = diabetes.target

# print(X.shape, Y.shape)
# Viewing the data in the form of a dataframe
X_df = pd.DataFrame(X, columns=diabetes.feature_names) 
X_df.describe()
# Implementing from scratch with our own functions
def add_bias_feature(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

def fit(X, y):
    X_with_bias = add_bias_feature(X)
    weights = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
    return weights

def predict(X, weights):
    X_with_bias = add_bias_feature(X) 
    return np.dot(X_with_bias, weights)

def calculate_mse(y_true, y_pred): 
    return np.mean((y_true - y_pred) ** 2)

def calculate_r_squared(y_true, y_pred): 
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y) ** 2) 
    ss_residual = np.sum((y_true - y_pred) ** 2) 
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def plotter(ax, y_true, y_pred):
    ax.plot([min(y_true), max(y_true)], [min(y_pred), max(y_pred)], linestyle='--', color='red', linewidth=2, label='Regression Line')

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Train the linear regression model 
weights = fit(X_train, y_train)

# Extract intercept and coefficients 
intercept = weights[0]
coefficients = weights[1:]

# Make predictions on the test set 
y_pred = predict(X_test, weights)

# # Calculate metrics
# mse = calculate_mse(y_test, y_pred)
# r_squared = calculate_r_squared(y_test, y_pred)
print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")

# print(f"Mean Squared Error: {mse}")
# print(f"R-squared (Coefficient of Determination): {r_squared}")

# Plotting predictions vs. actual values 
plt.scatter(y_test, y_pred) 
plt.xlabel("Actual Values") 
plt.ylabel("Predicted Values") 
plt.title("Actual vs. Predicted Values") 
plt.show()

# Plotting predictions vs. actual values with regression line 
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
plotter(ax, y_test, y_pred)
ax.set_xlabel("Actual Values") 
ax.set_ylabel("Predicted Values") 
ax.set_title("Actual vs. Predicted Values") 
ax.legend()
plt.show()