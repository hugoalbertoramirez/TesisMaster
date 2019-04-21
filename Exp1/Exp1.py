import numpy as np
from GM1D import GM1D

#np.random.seed(42)

K = 10

gaussian_params = []
for k in range(K):
    mean = np.random.rand() * 100
    var = (np.random.rand() + 1) * 3
    gaussian_params.append([mean, var])

gaussian_params = sorted(gaussian_params, key=lambda x: x[0])
gaussian_params = np.array(gaussian_params)
pi = np.random.sample(K) / 2
pi = pi / np.sum(pi)

gaussian_classes = np.array([0] * int(K / 2) + [1] * int(K / 2))


# gaussian_params = np.array([[7, 5], [50, 7], [80, 10], [120, 5], [150, 7]], dtype=float)
# gaussian_classes = np.array([0, 0, 0, 0, 1])
# pi = np.array([.3, .2, .1, .2, .2])
# samples = 10000

GM = GM1D(gaussian_params, pi, gaussian_classes)
GM.run(5000)
