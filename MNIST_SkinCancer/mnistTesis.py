import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import math
import cvxpy as cvx
from sklearn import svm, datasets
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import mixture
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

def plot_hist():
    sns.set()
    file = 'C:\\Users\HugoAlberto\Desktop\Git\TesisMaster\MNIST_SkinCancer\hmnist_8_8_L.csv'
    data = pd.read_csv(file)
    labels = ['Enfermedad de \nBowen',
            'Carcinoma de \ncélulas basales',
            'Lesiones benignas \nsimilares a la queratosis',
            'Dermatofibroma',
            'Melanoma',
            'Nevos melanocíticos',
            'Lesiones vasculares \ny Granulomas piógenos']

    ax = sns.countplot(y="label", color='c', data=data)
    plt.xlabel('Frecuencia')
    plt.ylabel('Diagnóstico')
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.show()


def build_dataset():
    file = 'C:\\Users\HugoAlberto\Desktop\Git\TesisMaster\MNIST_SkinCancer\hmnist_8_8_L.csv'
    data = pd.read_csv(file)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    class_label = 4

    X_train = np.array(train_set.iloc[:, 0:-1])
    y_train = np.array(train_set['label'])
    y_train = (y_train == class_label).astype(int)

    X_test = np.array(test_set.iloc[:, 0:-1])
    y_test = np.array(test_set['label'])
    y_test = (y_test == class_label).astype(int)

    return X_train, y_train, X_test, y_test


def build_gaussians(X_train, y_train, n_clusters):
    clf = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)

    clf.fit(X_train[y_train == 0])
    means_0, pi_0, covariances_0 = clf.means_, clf.weights_, clf.covariances_

    clf.fit(X_train[y_train == 1])
    means_1, pi_1, covariances_1 = clf.means_, clf.weights_, clf.covariances_

    means = np.concatenate((means_0, means_1), axis=0)
    covariances = np.concatenate((covariances_0, covariances_1), axis=0)
    pi_ = np.concatenate((pi_0, pi_1), axis=0)

    gaussians = list(zip(means, covariances, pi_))

    return gaussians, len(means_0), len(means_1)


def matrix_kernel(gaussians, y):
    k = len(y)

    kernel = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            kernel[i][j] = y[i] * y[j] * PPK(gaussians[i], gaussians[j])

    return kernel


def PPK(gaussian_1, gaussian_2):
    mean1 = gaussian_1[0]
    mean2 = gaussian_2[0]

    cov1 = gaussian_1[1]
    cov2 = gaussian_2[1]

    #print('>> 1 ', mean1, mean2, cov1, cov2, '1 <<')

    cov1_inv = inv(cov1)
    cov2_inv = inv(cov2)

    # print('>> 2', cov1_inv, cov2_inv, '2 <<')

    cov_dash = inv(cov1_inv + cov2_inv)
    mean_dash = np.matmul(cov1_inv, mean1) + np.matmul(cov2_inv, mean2)

    # print('>> 3', cov_dash, mean_dash, '3 <<')

    rho = 1
    D = 4
    pi = math.pi

    n = (2 * pi) ** ((1 - 2 * rho) * D / 2)
    n *= rho ** (-D / 2)
    n *= det(cov_dash) ** (1 / 2)
    n *= det(cov1) ** (-rho / 2)
    n *= det(cov2) ** (-rho / 2)

    # print('>> 4', n, '4 <<')

    arg = np.matmul(np.matmul(mean1.T, cov1_inv), mean1)
    arg += np.matmul(np.matmul(mean2.T, cov2_inv), mean2)
    arg -= np.matmul(np.matmul(mean_dash.T, cov_dash), mean_dash)
    arg *= (-rho / 2)
    if arg > 200:
        arg = 200

    # print('>> 5', arg, '5 <<')

    return n * math.exp(arg)


def print_kernel(kernel):
    for i in range(np.shape(kernel)[0]):
        for j in range(np.shape(kernel)[1]):
            print('%10.7s ' % (kernel[i][j]), end="")
        print()


def minimize(kernel, y, C, pi_):
    N = kernel.shape[0]

    a = cvx.Variable(N)
    obj = cvx.Minimize((1 / 2) * cvx.quad_form(a, kernel) - cvx.sum_entries(a))

    constraints = [a >= 0,
                   a <= cvx.mul_elemwise(C, pi_),
                   cvx.sum_entries(a.T * y) == 0]

    prob = cvx.Problem(obj, constraints)

    prob.solve()

    print("status:", prob.status)

    return [x[0, 0] for x in a.value]


def b_value(kernel, y, alpha):
    eps = 1e-6
    b = 0
    NS = 0

    for n in range(len(y)):
        if alpha[n] > eps:
            NS = NS + 1

            sum = 0
            for m in range(len(y)):
                if alpha[m] > eps:
                    sum += alpha[m] * y[m] + kernel[n][m]

            b += y[n] - sum

    b = (1 / NS) * b
    return b


def predict(mean_x, cov_x, alpha, y, gaussians, b):
    sum = 0
    N = len(y)
    gx = [mean_x, cov_x]

    for n in range(N):
        gn = gaussians[n]
        sum += alpha[n] * y[n] * PPK(gx, gn)

    if sum < 0:
        return 0
    else:
        return 1


def SVM_KKP(gaussians, len_y0, len_y1):
    y = [-1] * len_y0 + [1] * len_y1
    pi_ = [g[2] for g in gaussians]
    C = 1

    kernel = matrix_kernel(gaussians, y)
    # print_kernel(kernel)

    alphas = minimize(kernel, y, C, pi_)
    # print('a= ', alphas)

    b = b_value(kernel, y, alphas)
    # print('b=', b)

    return alphas, y, b


def find_metrics(X, Y, alphas, y, b, gaussians):
    y_pred = []

    for x_t, y_t in zip(X, Y):
        y_p = predict(x_t, np.identity(len(x_t)), alphas, y, gaussians, b)
        y_pred.append(y_p)

    return precision_score(Y, y_pred), recall_score(Y, y_pred), f1_score(Y, y_pred)


def plot_SVM(n_clusters, prec_test, recall_test, f1_test, train_times):
    sns.set()

    fig, ax1 = plt.subplots()

    plt.title('SVM con PPK')
    ax1.set_xlabel('Clusters')
    ax1.set_ylabel('score')
    ax1.set_ylim(0, 1.1)
    #plt.plot(n_clusters, prec_train, label='prec train')
    ax1.plot(n_clusters, prec_test, label='prec')
    ax1.plot(n_clusters, recall_test, label='recall')
    ax1.plot(n_clusters, f1_test, label='f1')
    ax1.legend(loc=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('segundos')
    #ax2.set_ylim(0, 0.11)
    ax2.grid(False)
    ax2.plot(n_clusters, train_times, sns.xkcd_rgb["dark green"], label='train time')
    ax2.legend(loc=4)
    plt.show()


X_train, y_train, X_test, y_test = build_dataset()
n_clusters = [2, 4, 6, 8, 10, 12, 14, 16]
#prec_train = []
prec_test = []
recall_test = []
f1_test = []
train_times = []

for n in n_clusters:
    start = time.time()

    gaussians, len_y0, len_y1 = build_gaussians(X_train, y_train, n_clusters=n)
    alphas, y, b = SVM_KKP(gaussians, len_y0, len_y1)

    end = time.time()
    trainingTime = end - start

    #p_train = precision(X_train, y_train, alphas, y, b, gaussians)
    prec_, recall_, f1_ = find_metrics(X_test, y_test, alphas, y, b, gaussians)

    #prec_train.append(p_train)
    prec_test.append(prec_)
    recall_test.append(recall_)
    f1_test.append(f1_)
    train_times.append(trainingTime)

plot_SVM(n_clusters, prec_test, recall_test, f1_test, train_times)