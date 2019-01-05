#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

i = 10

(countries, features, values) = a1.load_unicef_data()

N_TRAIN = 100

# Not normalized
x_train = values[:N_TRAIN,i]
x_test = values[N_TRAIN:,i]
t_train = values[0:N_TRAIN,1]
t_test = values[N_TRAIN:,1]

# Normalized
# x_train = a1.normalize_data(values[:N_TRAIN,7:])
# x_test = a1.normalize_data(values[N_TRAIN:,7:])
# t_train = a1.normalize_data(values[0:N_TRAIN,1])
# t_test = a1.normalize_data(values[N_TRAIN:,1])


# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py


# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)

# Evaluate regression on the linspace samples.
(w, tr_err) = a1.linear_regression(x_train, t_train, "ReLU", 0, 1)
(y_ev, te_err) = a1.evaluate_regression(np.asmatrix(x_ev).T, np.asmatrix(x_ev), w, "ReLU", 1)

print("---xxxx--->  The training error is: " + str(tr_err))
print("---xxxx--->  The testing error is: " + str(te_err))

# Produce a plot of results.
plt.plot(x_ev,y_ev, 'r.-', color = 'red', markersize = 1)
plt.plot(x_train,t_train,'bo', color = 'blue', markersize = 1)
plt.plot(x_test, t_test, 'bo', color = 'green', markersize = 1)
plt.title('Fit with ReLU, no regularization, not normalized')
plt.legend(['Random set','Training set', 'Test set'])
plt.ylabel(features[1])
plt.xlabel(features[i])
plt.show()

