#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

N_TRAIN = 100
train_err = {}
test_err = {}
i = 7
while i <= 14:
    # Not normalized
    x_train = values[:N_TRAIN,i]
    x_test = values[N_TRAIN:,i]
    t_train = values[0:N_TRAIN,1]
    t_test = values[N_TRAIN:,1]

    # TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
    d = 3

    (w, tr_err) = a1.linear_regression(x_train, t_train, "polynomial", 0, d)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, "polynomial", d)
    train_err[i+1] = tr_err
    test_err[i+1] = te_err
    i += 1

# Produce a plot of results.
plt.bar(train_err.keys(), train_err.values(), align = 'edge', width = .2)
plt.bar(test_err.keys(), test_err.values(), align = 'center', width = .2)
plt.ylabel('RMS')
plt.legend(['Train error','Test error'])
plt.title('Fit with 3 degree polynomial, no regularization')
plt.xlabel('Feature')
plt.show()
