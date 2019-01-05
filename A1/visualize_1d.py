#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

basis = 'polynomial' # basis polynomial or ReLU

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

# features 10, 11, 12
i = 12

N_TRAIN = 100;
# Select a single feature.
x_train = values[0:N_TRAIN,i]
t_train = targets[0:N_TRAIN]
x_test = values[N_TRAIN:,i]
t_test = targets[N_TRAIN:]


# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)


# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
(w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, 3)
(y_ev, te_err) = a1.evaluate_regression(np.asmatrix(x_ev).T, np.asmatrix(x_ev), w, basis, 3)

plt.plot(x_ev,y_ev, 'r.-', color = 'red', markersize = 1)
plt.plot(x_train,t_train,'bo', color = 'blue', markersize = 1)
plt.plot(x_test, t_test, 'bo', color = 'green', markersize = 1)
plt.title('A visualization of a regression estimate using random outputs')
plt.ylabel(features[1])
plt.legend(['Random set','Training set', 'Test set'])
plt.xlabel(features[i])
plt.show()
