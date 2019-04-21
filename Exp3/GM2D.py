import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture
import sklearn.datasets
from matplotlib.colors import LogNorm
from matplotlib import cm


class GM2D:
    def __init__(self, label='class 1', color='r', form='-', min_x=0, max_x=100, min_y=0, max_y=100):
        self.label = label
        self.color = color
        self.form = form
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        sns.set()
        np.random.seed(42)

    def create_from_data(self, gaussian_params, pi):
        self.gaussian_params = gaussian_params
        self.pi = pi
        self.K = len(pi)

    def create_random(self, min_x, max_x, min_y, max_y, K):
        gp, pi = self.create_random_params(min_x, max_x, min_y, max_y, K)
        self.create_from_data(gp, pi)
        return gp, pi

    def create_random_params(self, min_x, max_x, min_y, max_y, K):
        gaussian_params = []

        x_coor = [(max_x - min_x) * x + min_x for x in np.random.sample(K)]
        y_coor = [(max_y - min_y) * y + min_y for y in np.random.sample(K)]

        mean = list(zip(x_coor, y_coor))

        min_var, max_var = 10, 40
        for k in range(K):
            var_k = sklearn.datasets.make_spd_matrix(2)
            var_k = ((max_var - min_var) * var_k + min_var)

            gaussian_params.append([mean[k], var_k])

        pi = np.random.sample(K)
        pi = pi / np.sum(pi)

        # print values
        print('%20s %42s %20s' % ('mean', 'covar', 'pi'))
        for value in zip(gaussian_params, pi):
            print('%20s [%20s %20s, %20s' % (value[0][0][0], value[0][1][0][0], value[0][1][0][1], value[1]))
            print('%20s  %20s %20s]'      % (value[0][0][1], value[0][1][1][0], value[0][1][1][1]))
            print()

        return gaussian_params, pi

    def sample(self, size):
        sample_pi = np.random.choice(self.K, size=size, replace=True, p=self.pi)
        gp = self.gaussian_params

        x = [np.random.multivariate_normal(mean=gp[i][0], cov=gp[i][1]) for i in sample_pi]

        return x

    def draw_sample(self, X):
        # if self.label == 'class 1':
        #     self.grid = plt.subplot2grid((2, 3), (0, 0))
        # else:
        #     self.grid = plt.subplot2grid((2, 3), (1, 0))

        x = [i[0] for i in X]
        y = [i[1] for i in X]

        plt.scatter(x, y, c=self.color, marker=self.form, label=self.label)

        plt.xlim(self.min_x, self.max_x)
        plt.ylim(self.min_y, self.max_y)

    def pdf(self, X):
        y = np.zeros(len(X))
        for (mean, var), pi in zip(self.gaussian_params, self.pi):
            y += ss.multivariate_normal.pdf(X, mean=mean, cov=var) * pi
        return y

    def draw_pdf(self, X, alpha=1, shift_x=0, shift_y=0):
        xs = np.linspace(self.min_x, self.max_x, 300)
        ys = np.linspace(self.min_y, self.max_y, 300)

        X, Y = np.meshgrid(xs, ys)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = self.pdf(XX)
        Z = Z.reshape(X.shape)

        z_max = Z.max()
        z_min = Z.min()
        norm = cm.colors.Normalize(vmax=z_max, vmin=z_min)

        CS = plt.contour(X, Y, Z, levels=np.arange(z_min, z_max, (z_max - z_min) / 10), norm=norm)
        #CB = plt.colorbar(CS, shrink=1, extend='both')
        # ys = self.pdf(xs)
        # plt.plot(xs, ys, self.color + '-', label='Original MG ' + self.label, alpha=alpha)
        #
        # epsilon_x = (xs.max() - xs.min()) / 100
        # epsilon_y = (ys.max() - ys.min()) / 100
        #
        # for [mean, var], pi in zip(self.gaussian_params, self.pi):
        #     y_mean = self.pdf(mean)
        #
        #     plt.annotate(r'$\mu = ' + "{0:.2f}$\n".format(mean) +
        #                  '$\sigma^2 = ' + "{0:.2f}$\n".format(var) +
        #                  '$\pi = ' + "{0:.2f}$".format(pi),
        #                  xy=(mean, y_mean),
        #                  xytext=(mean + epsilon_x * (1 - shift_x), y_mean + epsilon_y * (1 - shift_y)),
        #                  color=self.color,
        #                  fontsize=10,
        #                  alpha=alpha)
        #     plt.plot([mean, mean], [0, y_mean], self.color, alpha=alpha)

    def GM_from_EM(self, X):
        clf = mixture.GaussianMixture(n_components=self.K, covariance_type='full')
        clf.fit(X.reshape(len(X), 1))

        means = clf.means_
        pi = clf.weights_
        covariances = np.sqrt(clf.covariances_)

        means = means[:, 0]
        covariances = covariances[:, 0, 0]
        gaussian_params = np.array(list(zip(means, covariances)))

        GM = GM2D(self.label + ' EM', color=self.color, form=self.form)
        GM.create_from_data(gaussian_params, pi)
        return GM

    def run(self, size=1000):
        sample = self.sample(size)
        self.draw_sample(sample)
        self.draw_pdf(sample)
        #
        # GM_EM = self.GM_from_EM(sample)
        # GM_EM.draw_pdf(sample, alpha=0.4, shift_x=10, shift_y=10)

        return sample#
        # , GM_EM