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


def nb_monomials_deg_dim2(deg):
    return(int(1+1.5*deg+0.5*deg*deg))

def lstsq(A, b, lambda_=0):
    return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ b)


# def heatmap(f, clip=5):
#     # example: heatmap(lambda x, y: x * x + y * y)
#     # clip: clip the function range to [-clip, clip] to generate a clean plot
#     #   set it to zero to disable this function
#
#     xx0 = xx1 = np.linspace(np.min(X), np.max(X), 72)
#     x0, x1 = np.meshgrid(xx0, xx1)
#     x0, x1 = x0.ravel(), x1.ravel()
#     z0 = f(x0, x1)
#
#     if clip:
#         z0[z0 > clip] = clip
#         z0[z0 < -clip] = -clip
#
#     plt.hexbin(x0, x1, C=z0, gridsize=50, cmap=cm.jet, bins=None)
#     plt.colorbar()
#     cs = plt.contour(
#         xx0, xx1, z0.reshape(xx0.size, xx1.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=cm.jet)
#     plt.clabel(cs, inline=1, fontsize=10)
#
#     pos = y[:] == +1.0
#     neg = y[:] == -1.0
#     plt.scatter(X[pos, 0], X[pos, 1], c='red', marker='+')
#     plt.scatter(X[neg, 0], X[neg, 1], c='blue', marker='v')
#     plt.show()

def heatmap(deg, w, clip=5):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    #   set it to zero to disable this function

    xx0 = xx1 = np.linspace(np.min(X), np.max(X), 72)
    x0, x1 = np.meshgrid(xx0, xx1)
    x0, x1 = x0.ravel(), x1.ravel()
    # z0 = f(x0, x1)
    X0 = np.array([x0, x1])
    X0_feat = assemble_feature(X0.T, deg)
    z0 = X0_feat @ w[deg, :nb_monomials_deg_dim2(deg)]


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
    plt.title('degree = {}'.format(deg))
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

def fit(X_, y_, split, lambda_):
    cut = int(np.floor(split*X_.shape[0]))
    train_x = X_[0:cut]
    train_y = y_[0:cut]
    valid_x = X_[cut:]
    valid_y = y_[cut:]

    # print(train_x.shape[0])
    # print(valid_x.shape[0])
    # print('--')

    w = lstsq(train_x, train_y, lambda_)
    Etrain = np.mean((train_y - train_x @ w)**2)
    Evalid = np.mean((valid_y - valid_x @ w)**2)

    return w, Etrain, Evalid

######

def question_a():
    area1 = np.abs(np.ma.masked_equal(y, -1))
    area2 = np.abs(np.ma.masked_equal(y, 1))

    plt.scatter(X[:,0], X[:,1], s=area1, marker='*')
    plt.scatter(X[:,0], X[:,1], s=area2, marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def question_b():
    heatmap_to_plot = [2, 4, 6, 8, 10, 12]

    w = np.zeros((max_deg, nb_monomials_deg_dim2(max_deg)))
    Etrain = np.zeros(max_deg)
    Evalid = np.zeros(max_deg)
    for deg in range(1, max_deg+1):
        global feat_x
        feat_x = assemble_feature(X, deg)
        bundle = np.c_[feat_x, y] #il est probable que le bundle/shuffle ne fonctionne pas bien (l'erreur de valisation ne devrait pas augmenter entre la version shuffled et non-shuffled)
        np.random.shuffle(bundle)
        feat_x_shuffled = bundle[:,:-1]
        y_shuffled = bundle[:, -1]
        w[deg-1, 0:nb_monomials_deg_dim2(deg)], Etrain[deg-1], Evalid[deg-1] = fit(feat_x_shuffled, y_shuffled, SPLIT, LAMBDA)

        ## plot in 1D a bit useless
        # y_predicted = (feat_x @ w[deg-1, 0:nb_monomials_deg_dim2(deg)])
        # plt.plot(y_predicted, 'o', linewidth = 0)
        # plt.plot(y, 'o', linewidth = 0)
        # plt.title(deg)
        # plt.show()

        ## plot in 2D heatmap
    # for deg in heatmap_to_plot:
        # heatmap(deg, w)

    plt.plot(range(1, max_deg+1), Etrain, label='Etrain')
    plt.plot(range(1, max_deg+1), Evalid, label='Evalid')
    plt.legend()
    plt.grid()
    plt.xlabel('degree of polynomial')
    plt.ylabel('average squared loss')
    plt.semilogy()
    plt.show()

    ## TODO: qu'est ce que c'est que cette histoire de heatmap...?

def question_c():
    nb_points = X.shape[0]
    split = 0.8
    cut = np.int(nb_points * split)
    bundle = np.c_[X, y]
    np.random.shuffle(bundle)
    X_shuffled = bundle[:,:-1]
    y_shuffled = bundle[:, -1]

    train_x = X_shuffled[0:cut]
    train_y = y_shuffled[0:cut]
    valid_x = X_shuffled[cut:]
    valid_y = y_shuffled[cut:]

    Etrain = np.zeros(max_deg)
    Evalid = np.zeros(max_deg)

    K = np.zeros((cut, cut))

    for deg in range(1, max_deg+1):
        print('degree = ', deg)
        for i in range(cut):
            for j in range(cut):
                K[i,j] = (1+train_x[i] @ train_x[j])**deg

        C = np.linalg.inv(K+LAMBDA*np.eye(cut)) @ train_y

        some_vect = np.zeros(cut)
        y_train_pred = np.zeros(cut)
        for i in range(cut):
            for j in range(cut):
                some_vect[j] = (1+train_x[j].T @ train_x[i])**deg
            y_train_pred[i] = some_vect @ C
        Etrain[deg-1] = np.mean((train_y - y_train_pred)**2)

        some_vect = np.zeros(cut)
        y_valid_pred = np.zeros(nb_points - cut)
        for i in range(nb_points - cut):
            for j in range(cut):
                some_vect[j] = (1+train_x[j].T @ valid_x[i])**deg
            y_valid_pred[i] = some_vect @ C
        Evalid[deg-1] = np.mean((valid_y - y_valid_pred)**2)

    plt.plot(range(1, max_deg+1), Etrain, label='Etrain')
    plt.plot(range(1, max_deg+1), Evalid, label='Evalid')
    plt.legend()
    plt.grid()
    plt.xlabel('degree of polynomial')
    plt.ylabel('average squared loss')
    plt.semilogy()
    plt.show()

def question_d():

    nb_splits = 20
    # degs = [5,6]
    degs = [5]
    # lambdas = [0.0001, 0.001, 0.01]
    lambdas = [0.0001]

    w = np.zeros((max_deg, nb_monomials_deg_dim2(max_deg)))
    Etrain = np.zeros((nb_splits, len(degs), len(lambdas)))
    Evalid = np.zeros((nb_splits, len(degs), len(lambdas)))

    for k in range(1, nb_splits):
        split_value = k/nb_splits
        for deg in degs:
            global feat_x
            feat_x = assemble_feature(X, deg)
            bundle = np.c_[feat_x, y]
            np.random.shuffle(bundle)
            feat_x_shuffled = bundle[:,:-1]
            y_shuffled = bundle[:, -1]
            for i in range(len(lambdas)):
                w[deg-degs[0], 0:nb_monomials_deg_dim2(deg)], Etrain[k, deg-degs[0], i], Evalid[k, deg-degs[0], i] = fit(feat_x_shuffled, y_shuffled, split_value, lambdas[i])


    for i in range(len(degs)):
        for k in range(len(lambdas)):
            plt.plot(np.linspace(1/nb_splits, 1, nb_splits-1), Etrain[1:,i,k], '-', label = 'Etrain (lambda = {})'. format(lambdas[k]))
            plt.plot(np.linspace(1/nb_splits, 1, nb_splits-1), Evalid[1:,i,k], '--', label = 'Evalid (lambda = {})'. format(lambdas[k]))
        plt.legend()
        plt.title('degree = {}'.format(degs[i]))
        plt.grid()
        plt.show()
    ## pour d = 5
    # CHELOU: l'erreur de training augmente
    # MOINS CHELOU: l'erreur de validation diminue (mais elle part de 2 -> chelou)
    ## pour d = 6
    # CHELOU: l'erreur de training augmente
    # MOINS CHELOU: l'erreur de validation diminue mais ne converge pas (mais elle part de 2 -> chelou)



def main():
    # example usage of heatmap
    # heatmap(lambda x0, x1: x0 * x0 + x1 * x1)
    # question_a()
    # question_b()
    # question_c()
    question_d()







if __name__ == "__main__":
    main()
