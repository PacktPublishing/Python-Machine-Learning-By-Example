import numpy as np


# Gradient descent based linear regression from scratch
def compute_prediction(X, weights):
    """ Compute the prediction y_hat based on current weights
    Args:
        X (numpy.ndarray)
        weights (numpy.ndarray)
    Returns:
        numpy.ndarray, y_hat of X under weights
    """
    predictions = np.dot(X, weights)
    return predictions

def update_weights_gd(X_train, y_train, weights, learning_rate):
    """ Update weights by one step
    Args:
        X_train, y_train (numpy.ndarray, training data set)
        weights (numpy.ndarray)
        learning_rate (float)
    Returns:
        numpy.ndarray, updated weights
    """
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def compute_cost(X, y, weights):
    """ Compute the cost J(w)
    Args:
        X, y (numpy.ndarray, data set)
        weights (numpy.ndarray)
    Returns:
        float
    """
    predictions = compute_prediction(X, weights)
    cost = np.mean((predictions - y) ** 2 / 2.0)
    return cost

def train_linear_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
    """ Train a linear regression model with gradient descent
    Args:
        X_train, y_train (numpy.ndarray, training data set)
        max_iter (int, number of iterations)
        learning_rate (float)
        fit_intercept (bool, with an intercept w0 or not)
    Returns:
        numpy.ndarray, learned weights
    """
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_rate)
        # Check the cost for every 100 (for example) iterations
        if iteration % 100 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)


# A small example
X_train = np.array([[6], [2], [3], [4], [1], [5], [2], [6], [4], [7]])

y_train = np.array([5.5, 1.6, 2.2, 3.7, 0.8, 5.2, 1.5, 5.3, 4.4, 6.8])

weights = train_linear_regression(X_train, y_train, max_iter=100, learning_rate=0.01, fit_intercept=True)

X_test = np.array([[1.3], [3.5], [5.2], [2.8]])

predictions = predict(X_test, weights)

import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], y_train, marker='o', c='b')
plt.scatter(X_test[:, 0], predictions, marker='*', c='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# The diabetes example
from sklearn import datasets
diabetes = datasets.load_diabetes()
print(diabetes.data.shape)

num_test = 30    # the last 30 samples as testing set
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]

weights = train_linear_regression(X_train, y_train, max_iter=5000, learning_rate=1, fit_intercept=True)

X_test = diabetes.data[-num_test:, :]
y_test = diabetes.target[-num_test:]

predictions = predict(X_test, weights)

print(predictions)
print(y_test)






# Directly use SGDRegressor from scikit-learn
from sklearn.linear_model import SGDRegressor
regressor = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, learning_rate='constant', eta0=0.01, n_iter=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)
print(regressor.score(X_test, y_test))


# Measuring model performance after hyperparameter tuning with grid search
diabetes = datasets.load_diabetes()
num_test = 30    # the last 30 samples as testing set
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
X_test = diabetes.data[-num_test:, :]
y_test = diabetes.target[-num_test:]

param_grid = {
    "alpha": [1e-07, 1e-06, 1e-05],
    "penalty": [None, "l2"],
    "eta0": [0.001, 0.005, 0.01],
    "n_iter": [300, 1000, 3000]
}

from sklearn.model_selection import GridSearchCV
regressor = SGDRegressor(loss='squared_loss', learning_rate='constant')
grid_search = GridSearchCV(regressor, param_grid, cv=3)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
regressor_best = grid_search.best_estimator_
# regressor_best.score(X_test, y_test)

predictions = regressor_best.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mean_squared_error(y_test, predictions)
mean_absolute_error(y_test, predictions)
r2_score(y_test, predictions)

