import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import mixture
import seaborn as sns


class GM1D:
    def __init__(self, gaussian_params, pi, gaussian_classes):
        self.gaussian_params = gaussian_params
        self.pi = pi
        self.gaussian_classes = gaussian_classes
        self.K = gaussian_params.shape[0]

        sns.set()
        np.random.seed(42)

    def sample(self, size):
        gp = self.gaussian_params
        pi = self.pi
        K = self.K

        sample_pi = np.random.choice(K, size=size, replace=True, p=pi)

        x = np.fromiter((ss.norm.rvs(*(gp[i])) for i in sample_pi), dtype=np.float32)

        return x, sample_pi

    def pdf(self, x):
        y = np.zeros_like(x)

        for (miu, var), pi in zip(self.gaussian_params, self.pi):
            y += ss.norm.pdf(x, loc=miu, scale=var) * pi

        return y

    def pdfWithParams(self, x, means, covariances, pi):
        y = np.zeros_like(x)

        means = means[:, 0]
        covariances = covariances[:, 0, 0]
        gaussian_params = np.array(list(zip(means, covariances)))

        for (miu, var), pi in zip(gaussian_params, pi):
            y += ss.norm.pdf(x, loc=miu, scale=var) * pi

        return y

    def run(self, size=10000):
        x, sample_pi = self.sample(size)
        y = self.gaussian_classes[sample_pi]

        # draw points of each class:
        xs = np.linspace(x.min(), x.max(), 300)
        ys = self.pdf(xs)

        class1 = []
        class2 = []
        for i in range(x.size):
            if self.gaussian_classes[sample_pi[i]] == 0:
                class1.append(x[i])
            else:
                class2.append(x[i])

        plt.plot(class1, np.ones(len(class1)) * ys.max() * 0.01, 'ro', markersize=3, label='class 1')
        plt.plot(class2, np.ones(len(class2)) * ys.max() * 0.02, 'bo', markersize=3, label='class 2')
        plt.hist(x, density=True, bins="fd", color="skyblue")
        plt.xlabel("x")
        plt.ylabel("MG(x)")

        # draw original mixture of gaussian:
        plt.plot(xs, ys, 'b', label='Original MG', )

        epsilonX = (xs.max() - xs.min()) / 100
        epsilonY = (ys.max() - ys.min()) / 100

        for [mean, var], pi in zip(self.gaussian_params, self.pi):
            y_mean = self.pdf(mean)

            plt.annotate(r'$\mu = $' + "{0:.2f}\n".format(mean) +
                         '$\sigma^2 = $' + "{0:.2f}\n".format(var) +
                         '$\pi = $' + "{0:.2f}".format(pi),
                         xy=(mean, y_mean),
                         xytext=(mean, y_mean + epsilonY))
            plt.plot([mean, mean], [0, y_mean], 'b')

        # draw calculated mixture of gaussians:
        clf_GM = mixture.GaussianMixture(n_components=len(self.pi), covariance_type='full')
        clf_GM.fit(x.reshape(len(x), 1))

        means_GM = clf_GM.means_
        pi_GM = clf_GM.weights_
        var_GM = np.sqrt(clf_GM.covariances_)

        ys_MG = self.pdfWithParams(xs, means_GM, var_GM, pi_GM)
        plt.plot(xs, ys_MG, 'purple', label='MG from EM')

        for mean, var, pi in zip(means_GM, var_GM, pi_GM):
            mean = mean[0]
            var = var[0, 0]

            y_mean = self.pdfWithParams(mean, means_GM, var_GM, pi_GM)
            # y_mean = clf_GM.predict_proba(mean.reshape(1, -1))

            plt.annotate(r'$\mu = $' + "{0:.2f}\n".format(mean) +
                         '$\sigma^2 = $' + "{0:.2f}\n".format(var) +
                         '$\pi = $' + "{0:.2f}".format(pi),
                         xy=(mean, y_mean),
                         xytext=(mean + epsilonX, y_mean - 10 * epsilonY))
            plt.plot([mean, mean], [0, y_mean], 'purple')

        self.svm(x, y, ys.max())

    def svm(self, X, y, height):

        # train with all points
        X = X.reshape(len(X), 1)
        clf = svm.SVC(kernel='linear', C=1e3)
        clf.fit(X, y)

        w = clf.coef_[0, 0]
        b = clf.intercept_[0]
        p0 = -b / w

        plt.plot([p0, p0], [0, height], 'b')

        # train with only means and weights
        means = self.gaussian_params[:, 0]
        X_means = means.reshape(len(means), 1)
        y_means = self.gaussian_classes

        weights = self.pi

        clf_weights = svm.SVC(kernel='linear', C=1e3)
        clf_weights.fit(X_means, y_means, sample_weight=weights)

        w = clf_weights.coef_[0, 0]
        b = clf_weights.intercept_[0]
        p0 = -b / w

        plt.plot([p0, p0], [0, height], 'purple')

        # train only with means and weights and producto kernel

        plt.legend(loc='upper right')
        plt.show()

        # plt.savefig('xlim_and_ylim.png')
