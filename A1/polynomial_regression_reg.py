#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

N_TRAIN = 100 # number of training points
Normalized = True # Normalized input true/false
basis = 'polynomial' # basis polynomial or ReLU
d = 2 # number of polynomial degrees
train_err = {} # train dictionary initialized
test_err = {} # test dictionary initialized
reg_lambda = [0, 0.01, 0.1, 1, 100, 1000, 10000] # lambda parameter in normalization

j = 0  # lambda_reg loop initialized
while j <= 6:
    print(reg_lambda[j])
    train_err[reg_lambda[j]] = 0
    test_err[reg_lambda[j]] = 0
    k = 0 #initialize cross validation
    while k < 10:

        if (k == 9):
            g = 10
        else:
            g = 0

        if k == 0:
            x_train = values[10*(k+1):N_TRAIN-g,7:]
            x_test = values[10*k:10*(k+1),7:]
            t_train = values[10*(k+1):N_TRAIN-g,1]
            t_test = values[10*k:10*(k+1),1]
        else:
            x_train = np.vstack((values[10 * (k + 1):N_TRAIN - g, 7:],values[:10 * k, 7:]))
            x_test = values[10 * k:10 * (k + 1), 7:]
            t_train = np.vstack((values[10 * (k + 1):N_TRAIN - g, 1],values[:10 * k, 1]))
            t_test = values[10 * k:10 * (k + 1), 1]

        if (Normalized):
            x_train = a1.normalize_data(x_train)
            x_test = a1.normalize_data(x_test)
            t_train = a1.normalize_data(t_train)
            t_test = a1.normalize_data(t_test)
        else:
            ""
        # TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
        (w, tr_err) = a1.linear_regression(x_train, t_train, basis, reg_lambda[j], d)
        (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, basis, d)


        train_err[reg_lambda[j]] = train_err[reg_lambda[j]] + tr_err/10
        test_err[reg_lambda[j]] = test_err[reg_lambda[j]] + te_err/10

        k += 1
    j += 1

# Produce a plot of results.
plt.semilogx(test_err.keys(), test_err.values())
# plt.bar(train_err.keys(), train_err.values(), align = 'edge', width = 0.2)
plt.ylabel('Average RMS')
# plt.legend(['Test error','Tsraining error'])
plt.title('Validation RMS with 2 degree polynomial, lambda regularization')
plt.xlabel('Lambda')
plt.show()
