from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
from numpy.linalg import inv
from numpy.linalg import det
import math
import cvxpy as cvx

class SVM1D:

    def train_w_all_points(self, X1, X2):
        X = np.concatenate((X1, X2))
        y = np.array([0] * len(X1) + [1] * len(X2))

        X = X.reshape(len(X), 1)
        clf = svm.SVC(kernel='linear')
        clf.fit(X, y)

        w = clf.coef_[0, 0]
        b = clf.intercept_[0]
        p0 = -b / w

        _, ymax = ylim()

        plt.plot([p0, p0], [0, ymax], color='black', label='svm (trained w/all data)')

    def train_w_means_pi(self, GM1, GM2):
        gp1, pi1 = GM1.gaussian_params, GM1.pi
        gp2, pi2 = GM2.gaussian_params, GM2.pi

        means = np.concatenate((gp1[:, 0], gp2[:, 0]))
        pi = np.concatenate((pi1, pi2))

        X = means.reshape(len(means), 1)
        y = np.array([0] * len(gp1) + [1] * len(gp2))

        clf = svm.SVC(kernel='linear', C=1e3)
        clf.fit(X, y, sample_weight=pi)

        w = clf.coef_[0, 0]
        b = clf.intercept_[0]
        p0 = -b / w

        _, ymax = ylim()

        plt.plot([p0, p0], [0, ymax], color='purple', label='svm (trained w/means and weights)')

    def svm_train_w_pdf(self, GM1, GM2, min, max):
        gaussian_params = np.concatenate((GM1.gaussian_params, GM2.gaussian_params))
        y = [-1] * len(GM1.gaussian_params) + [1] * len(GM2.gaussian_params)
        C = 10000

        kernel = self.buildKernel(gaussian_params, y)
        alpha = self.minimize(kernel, y, C)
        b = self.b_value(kernel, y, alpha, C)

        print('kernel')
        self.print_kernel(kernel)
        print('a', alpha)
        print('b', b)

        _, ymax = ylim()

        # evaluate points:
        xs = np.linspace(min, max, 300)
        ys = []
        for x in xs:
            aux = self.predict(x, alpha, y, gaussian_params, b)
            ys.append(aux * ymax)

        plt.plot(xs, ys, color='purple', label='svm (trained w/kernel prod. of gauss.)')

    def buildKernel(self, gaussian_params, y):
        k = len(y)

        kernel = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                factor = y[i] * y[j]
                kernel[i][j] = factor * self.product_gaussians_kernel(gaussian_params[i],
                                                                      gaussian_params[j])

        return kernel

    def product_gaussians_kernel(self, gaussian_params_1, gaussian_params_2):
        mean1 = gaussian_params_1[0]
        mean2 = gaussian_params_2[0]

        var1 = gaussian_params_1[1]
        var2 = gaussian_params_2[1]

        sigma1 = np.array([var1])
        sigma2 = np.array([var2])

        sigma_inv1 = np.array([1 / var1]) #inv(sigma1)
        sigma_inv2 = np.array([1 / var2]) #inv(sigma2)

        sigma_cruz = 1 / (sigma_inv1 + sigma_inv2) # inv(sigma_inv1 + sigma_inv2)
        mean_cruz = sigma_inv1 * mean1 + sigma_inv2 * mean2

        rho = 0.5
        D = 1
        pi = math.pi

        n = (2 * pi) ** ((1 - 2 * rho) * D / 2)
        n *= rho ** (-D / 2)
        n *= sigma_cruz ** (1 / 2)
        n *= sigma1 ** (-rho / 2)
        n *= sigma2 ** (-rho / 2)

        arg = mean1 * sigma_inv1 * mean1
        arg += mean2 * sigma_inv2 * mean2
        arg -= mean_cruz * sigma_cruz * mean_cruz
        arg *= (-rho / 2)

        return n * math.exp(arg)

    def print_kernel(self, kernel):
        for i in range(np.shape(kernel)[0]):
            for j in range(np.shape(kernel)[1]):
                print('%10.7s ' % (kernel[i][j]), end="")
            print()

    def minimize(self, kernel, y, C):
        N = kernel.shape[0]

        a = cvx.Variable(N)
        m1 = np.array([-1] * N)
        obj = cvx.Minimize((1 / 2) * cvx.quad_form(a, kernel) + m1.T * a)

        constraints = [a >= 0,
                       a <= C,
                       cvx.sum_entries(a.T * y) == 0]

        prob = cvx.Problem(obj, constraints)

        prob.solve()

        print("status:", prob.status)

        return [x[0, 0] for x in a.value]

    def b_value(self, kernel, y, alpha, C):
        eps = 1e-6

        b = 0
        M = 0
        for n in range(len(y)):
            if 0 <= alpha[n] and alpha[n] <= C:
                M = M + 1

                sum = 0
                for m in range(len(y)):
                    if alpha[m] > 0:
                        sum += alpha[m] * y[m] + kernel[n][m]

                b += y[n] - sum

        b = (1 / M) * b
        return b

    def b_value__(self, kernel, y, alpha, C):
        eps = 1e-6

        NS = 0
        sum_ext = 0
        for n in range(len(alpha)):
            if alpha[n] > 0:
                NS = NS + 1

                sum_int = 0
                for m in range(len(alpha)):
                    if alpha[m] > 0:
                        sum_int += alpha[m] * y[m] + kernel[n][m]

                sum_ext += y[n] - sum_int

        b = (1.0 / NS) * sum_ext

        return b

    def b_value__(self, kernel, y, alpha, C):
        eps = 1e-6
        index_S = 1

        sum_int = 0
        for m in range(len(y)):
            if alpha[m] > 0:
                sum_int += alpha[m] * y[m] + kernel[index_S][m]

        b = y[index_S] - sum_int

        return b

    def predict(self, x, alpha, y, gaussian_params, b):
        sum = 0
        N = len(y)
        gp_x = [x, 1]

        for n in range(N):
            gp_n = gaussian_params[n]

            sum += alpha[n] * y[n] * self.product_gaussians_kernel(gp_n, gp_x)

        #return sum
        return np.sign(sum)






