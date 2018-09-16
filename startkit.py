#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# choose the data you want to load
data = np.load('circle.npz')
# data = np.load('heart.npz')
# data = np.load('asymmetric.npz')

SPLIT = 0.8
X = data["x"]
y = data["y"]
X /= np.max(X)  # normalize the data

n_train = int(X.shape[0] * SPLIT)
X_train = X[:n_train:, :]
X_valid = X[n_train:, :]
y_train = y[:n_train]
y_valid = y[n_train:]

LAMBDA = 0.001

max_deg = 16


def lstsq(A, b, lambda_=0):
    return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ b)


def heatmap(f, clip=5):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    #   set it to zero to disable this function

    xx0 = xx1 = np.linspace(np.min(X), np.max(X), 72)
    x0, x1 = np.meshgrid(xx0, xx1)
    x0, x1 = x0.ravel(), x1.ravel()
    z0 = f(x0, x1)

    if clip:
        z0[z0 > clip] = clip
        z0[z0 < -clip] = -clip

    plt.hexbin(x0, x1, C=z0, gridsize=50, cmap=cm.jet, bins=None)
    plt.colorbar()
    cs = plt.contour(
        xx0, xx1, z0.reshape(xx0.size, xx1.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=cm.jet)
    plt.clabel(cs, inline=1, fontsize=10)

    pos = y[:] == +1.0
    neg = y[:] == -1.0
    plt.scatter(X[pos, 0], X[pos, 1], c='red', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='blue', marker='v')
    plt.show()

def assemble_feature(x, D):
    n_feature = x.shape[1]
    Q = [(np.ones(x.shape[0]), 0, 0)]
    i = 0
    while Q[i][1] < D:
        cx, degree, last_index = Q[i]
        for j in range(last_index, n_feature):
            Q.append((cx * x[:, j], degree + 1, j))
        i += 1
    return np.column_stack([q[0] for q in Q])


def fit(X):
    # Etrain = 0
    # Evalid = 0
    cut = int(np.floor(SPLIT*X.shape[0]))
    train_x = X[0:cut]
    train_y = y[0:cut]
    valid_x = X[cut:]
    valid_y = y[cut:]

    w = lstsq(train_x, train_y, lambda_=LAMBDA)
    Etrain = np.mean((train_y - train_x @ w)**2)
    Evalid = np.mean((valid_y - valid_x @ w)**2)

    return np.mean(Etrain), np.mean(Evalid)
    # return w, np.mean(Etrain), np.mean(Evalid)


def question_a():
    area1 = np.abs(np.ma.masked_equal(y, -1))
    area2 = np.abs(np.ma.masked_equal(y, 1))

    plt.scatter(X[:,0], X[:,1], s=area1, marker='*')
    plt.scatter(X[:,0], X[:,1], s=area2, marker='o')
    # plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def question_b():
    # w = np.zeros([max_deg])
    Etrain = np.zeros(max_deg)
    Evalid = np.zeros(max_deg)
    for deg in range(1, max_deg+1):
        global feat_x
        feat_x = assemble_feature(X, deg)
        Etrain[deg-1], Evalid[deg-1] = fit(feat_x)

    plt.plot(range(1, max_deg+1), Etrain, label='Etrain')
    plt.plot(range(1, max_deg+1), Evalid, label='Evalid')
    plt.legend()
    plt.grid()
    plt.xlabel('degree of polynomial')
    plt.ylabel('average squared loss')
    plt.show()

def main():
    # example usage of heatmap
    # heatmap(lambda x0, x1: x0 * x0 + x1 * x1)
    # question_a()
    question_b()







if __name__ == "__main__":
    main()
