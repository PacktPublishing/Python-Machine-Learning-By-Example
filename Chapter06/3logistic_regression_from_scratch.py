import numpy as np
import matplotlib.pyplot as plt

def sigmoid(input):
    return 1.0 / (1 + np.exp(-input))



# Gradient descent based logistic regression from scratch
def compute_prediction(X, weights):
    """ Compute the prediction y_hat based on current weights
    Args:
        X (numpy.ndarray)
        weights (numpy.ndarray)
    Returns:
        numpy.ndarray, y_hat of X under weights
    """
    z = np.dot(X, weights)
    predictions = sigmoid(z)
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
    cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    return cost

def train_logistic_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
    """ Train a logistic regression model
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
        if iteration % 1000 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)


# A example
X_train = np.array([[6, 7],
                    [2, 4],
                    [3, 6],
                    [4, 7],
                    [1, 6],
                    [5, 2],
                    [2, 0],
                    [6, 3],
                    [4, 1],
                    [7, 2]])

y_train = np.array([0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1])

weights = train_logistic_regression(X_train, y_train, max_iter=1000, learning_rate=0.1, fit_intercept=True)

X_test = np.array([[6, 1],
                   [1, 3],
                   [3, 1],
                   [4, 5]])

predictions = predict(X_test, weights)

plt.scatter(X_train[:,0], X_train[:,1], c=['b']*5+['k']*5, marker='o')
colours = ['k' if prediction >= 0.5 else 'b' for prediction in predictions]
plt.scatter(X_test[:,0], X_test[:,1], marker='*', c=colours)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()









# Deploy logistic regression by gradient descent to click-through prediction

import csv

def read_ad_click_data(n, offset=0):
    X_dict, y = [], []
    with open('train.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i in range(offset):
            next(reader)
        i = 0
        for row in reader:
            i += 1
            y.append(int(row['click']))
            del row['click'], row['id'], row['hour'], row['device_id'], row['device_ip']
            X_dict.append(row)
            if i >= n:
                break
    return X_dict, y

n = 10000
X_dict_train, y_train = read_ad_click_data(n)

from sklearn.feature_extraction import DictVectorizer
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

X_dict_test, y_test = read_ad_click_data(n, n)
X_test = dict_one_hot_encoder.transform(X_dict_test)

X_train_10k = X_train
y_train_10k = np.array(y_train)

import timeit
start_time = timeit.default_timer()
weights = train_logistic_regression(X_train_10k, y_train_10k, max_iter=10000, learning_rate=0.01, fit_intercept=True)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))

X_test_10k = X_test

predictions = predict(X_test_10k, weights)
from sklearn.metrics import roc_auc_score
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test, predictions)))



n = 100000
X_dict_train, y_train = read_ad_click_data(n)
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

X_train_100k = X_train
y_train_100k = np.array(y_train)

start_time = timeit.default_timer()
weights = train_logistic_regression(X_train_100k, y_train_100k, max_iter=10000, learning_rate=0.01, fit_intercept=True)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))





def update_weights_sgd(X_train, y_train, weights, learning_rate):
    """ One weight update iteration: moving weights by one step based on each individual sample
    Args:
        X_train, y_train (numpy.ndarray, training data set)
        weights (numpy.ndarray)
        learning_rate (float)
    Returns:
        numpy.ndarray, updated weights
    """
    for X_each, y_each in zip(X_train, y_train):
        prediction = compute_prediction(X_each, weights)
        weights_delta = X_each.T * (y_each - prediction)
        weights += learning_rate * weights_delta
    return weights

def train_logistic_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
    """ Train a logistic regression model
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
        weights = update_weights_sgd(X_train, y_train, weights, learning_rate)
        # Check the cost for every 2 (for example) iterations
        if iteration % 2 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights


# Train the SGD model based on 10000 samples
start_time = timeit.default_timer()
weights = train_logistic_regression(X_train_10k, y_train_10k, max_iter=5, learning_rate=0.01, fit_intercept=True)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))
predictions = predict(X_test_10k, weights)
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test, predictions)))


# Train the SGD model based on 100000 samples
start_time = timeit.default_timer()
weights = train_logistic_regression(X_train_100k, y_train_100k, max_iter=5, learning_rate=0.01, fit_intercept=True)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))

# Examine the performance on the next 10000 samples
X_dict_test, y_test_next10k = read_ad_click_data(10000, 100000)
X_test_next10k = dict_one_hot_encoder.transform(X_dict_test)
predictions = predict(X_test_next10k, weights)
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test_next10k, predictions)))
