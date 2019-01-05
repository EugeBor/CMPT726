#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

N_TRAIN = 100 # number of training points
Normalized = True # Normalized input true/false
basis = 'polynomial' # basis polynomial or ReLU
d = 6 # number of polynomial degrees
i = 1 # loop initialized
train_err = {} # train dictionary initialized
test_err = {} # test dictionary initialized

x_train = values[:N_TRAIN,7:]
x_test = values[N_TRAIN:,7:]
t_train = values[0:N_TRAIN,1]
t_test = values[N_TRAIN:,1]

if (Normalized):
    x_train = a1.normalize_data(values[:N_TRAIN,7:])
    x_test = a1.normalize_data(values[N_TRAIN:,7:])
    t_train = a1.normalize_data(values[0:N_TRAIN,1])
    t_test = a1.normalize_data(values[N_TRAIN:,1])
else:
    ""

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

while i <= d:
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, i)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, basis, i)
    train_err[i] = tr_err
    test_err[i] = te_err
    i += 1



# Produce a plot of results.
plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, no regularization, normalized')
plt.xlabel('Polynomial degree')
plt.show()
