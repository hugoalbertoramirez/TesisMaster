import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture


class GM1D:
    def __init__(self, label='class 1', color='r', form='o'):
        self.label = label
        self.color = color
        self.form = form

        sns.set()
        np.random.seed(42)

    def create_from_data(self, gaussian_params, pi):
        self.gaussian_params = gaussian_params
        self.pi = pi
        self.K = len(pi)

    def create_random(self, min, max, K):
        gp, pi = self.create_random_params(max, min, K)
        self.create_from_data(gp, pi)
        return gp, pi

    def create_random_params(self, min, max, K=3):
        gaussian_params = []
        for k in range(K):
            mean = (max - min) * np.random.rand() + min
            var = (100 - 50) * (np.random.rand()) + 50
            gaussian_params.append([mean, var])

        gaussian_params = sorted(gaussian_params, key=lambda x: x[0])
        gaussian_params = np.array(gaussian_params)

        pi = np.random.sample(K)
        pi = pi / np.sum(pi)

        # print values
        print('%20s %20s %20s' % ('mean', 'var', 'pi'))
        for value in zip(gaussian_params, pi):
            print('%20s %20s %20s' % (value[0][0], value[0][1], value[1]))

        return gaussian_params, pi

    def sample(self, size):
        sample_pi = np.random.choice(self.K, size=size, replace=True, p=self.pi)
        x = np.fromiter((ss.norm.rvs(*(self.gaussian_params[i])) for i in sample_pi), dtype=np.float32)
        return x

    def draw_sample(self, X):
        y, _, _ = plt.hist(X, density=True, bins="fd", color="skyblue")
        plt.plot(X, np.ones(len(X)) * y.max() * 0.01, self.color + self.form, markersize=3, label=self.label)

    def pdf(self, X):
        y = np.zeros_like(X)
        for (mean, var), pi in zip(self.gaussian_params, self.pi):
            y += ss.norm.pdf(X, loc=mean, scale=var) * pi
        return y

    def draw_pdf(self, X, alpha=1, shift_x=0, shift_y=0):
        xs = np.linspace(X.min(), X.max(), 300)
        ys = self.pdf(xs)
        plt.plot(xs, ys, self.color + '-', label='Original MG ' + self.label, alpha=alpha)

        epsilon_x = (xs.max() - xs.min()) / 100
        epsilon_y = (ys.max() - ys.min()) / 100

        for [mean, var], pi in zip(self.gaussian_params, self.pi):
            y_mean = self.pdf(mean)

            plt.annotate(r'$\mu = ' + "{0:.2f}$\n".format(mean) +
                         '$\sigma^2 = ' + "{0:.2f}$\n".format(var) +
                         '$\pi = ' + "{0:.2f}$".format(pi),
                         xy=(mean, y_mean),
                         xytext=(mean + epsilon_x * (1 - shift_x), y_mean + epsilon_y * (1 - shift_y)),
                         color=self.color,
                         fontsize=10,
                         alpha=alpha)
            plt.plot([mean, mean], [0, y_mean], self.color, alpha=alpha)

    def GM_from_EM(self, X):
        clf = mixture.GaussianMixture(n_components=self.K, covariance_type='full')
        clf.fit(X.reshape(len(X), 1))

        means = clf.means_
        pi = clf.weights_
        covariances = np.sqrt(clf.covariances_)

        means = means[:, 0]
        covariances = covariances[:, 0, 0]
        gaussian_params = np.array(list(zip(means, covariances)))

        GM = GM1D(self.label + ' EM', color=self.color, form=self.form)
        GM.create_from_data(gaussian_params, pi)
        return GM

    def run(self, size=1000):
        sample = self.sample(size)
        self.draw_sample(sample)
        self.draw_pdf(sample)

        GM_EM = self.GM_from_EM(sample)
        GM_EM.draw_pdf(sample, alpha=0.4, shift_x=10, shift_y=10)

        return sample, GM_EM