#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Batch size
size = 20


# Load data.
data = np.genfromtxt('data.txt')

# For plotting data
# class1 = np.where(t == 0)
# X1 = X[class1]
# class2 = np.where(t == 1)
# X2 = X[class2]


# Error values over all iterations.
e_all = [[], [], [], [], []]

DATA_FIG = 1

# Set up the slope-intercept figure
SI_FIG = 2
plt.figure(SI_FIG)
plt.rcParams.update({'font.size': 15})
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])

# Step size for gradient descent.
eta_vars = [0.5, 0.3, 0.1, 0.05, 0.01]
i = 0

while i < 5:

    eta = eta_vars[i]

    # Initialize w.
    w = np.array([0.1, 0, 0])

    # Data matrix, with column of ones at end.
    X = data[:, 0:3]
    # Target values, 0 for class 1, 1 for class 2.
    t = data[:, 3]

    y = sps.expit(np.dot(X, w))

    err = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
    e_all[i].append(err)

    for iter in range(0, max_iter):

        j = 0

        np.random.shuffle(data)

        # Data matrix, with column of ones at end.
        X = data[:, 0:3]
        # Target values, 0 for class 1, 1 for class 2.
        t = data[:, 3]



        for j in range(0,int(len(t)/size)):
            # Compute output using current w on all data X.
            y = sps.expit(np.dot(X, w))

            # e is the error, negative log-likelihood (Eqn 4.90)
            e = -np.sum(np.multiply(t[j*size:size*(j+1)], np.log(y[j*size:size*(j+1)])) + np.multiply((1 - t[j*size:size*(j+1)]), np.log(1 - y[j*size:size*(j+1)])))/size

            # Gradient of the error, using Eqn 4.91
            grad_e = np.mean(np.multiply((y[j*size:size*(j+1)] - t[j*size:size*(j+1)]), X[j*size:size*(j+1)].T), axis=1)

            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w_old = w
            w = w - eta * grad_e

            # Add next step of separator in m-b space.
            plt.figure(SI_FIG)
            a2.plot_mb(w, w_old)

            err = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

            # Add this error to the end of error vector.
            if j == int(len(t)/size)-1:
                e_all[i].append(err)

            j += 1

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(err - e_all[i][iter - 1]) < tol:
                break


    i += 1




# Plot error over iterations
plt.figure()
plt.plot(e_all[0], label="0.5")
plt.plot(e_all[1], label="0.3")
plt.plot(e_all[2], label="0.1")
plt.plot(e_all[3], label="0.05")
plt.plot(e_all[4], label="0.01")
plt.legend(loc = "upper right", title="Step Size")
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression (SGD, Batch Size ' + str(size) + ')')
plt.xlabel('Epoch')
plt.show()
