import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
sys.path.append('C:\\Users\\HugoAlberto\Desktop\Git\Tesis\Exp3')
from GM2D import GM2D
#from SVM1D import SVM1D

GM1 = GM2D('class 1', color='r', form='o', min_x=0, max_x=100, min_y=0, max_y=100)
GM2 = GM2D('class 2', color='b', form='*', min_x=0, max_x=100, min_y=0, max_y=100)

GM1.create_random(min_x=10, max_x=70, min_y=10, max_y=70, K=3)
GM2.create_random(min_x=40, max_x=90, min_y=40, max_y=90, K=3)

# GM1.create_from_data(gaussian_params=[
#                        [(10, 10), np.array([[1, 0], [0, 1]], dtype=np.float32)],
#                        [(20, 20), np.array([[5, 8], [8, 13]], dtype=np.float32)]
#                       ],
#                       pi=[.9, .1])
# GM2.create_from_data(gaussian_params=[
#                        [(30, 40), np.array([[3, 4], [4, 4]], dtype=np.float32)],
#                        [(40, 50), np.array([[2, 9], [9, 4]], dtype=np.float32)]
#                       ],
#                       pi=[.5, .5])

plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

X1 = GM1.run(1000) # GM_EM1
X2 = GM2.run(1000)
#
# svm = SVM1D()
#
# svm.train_w_all_points(X1, X2)
# svm.train_w_means_pi(GM_EM1, GM_EM2)
# svm.svm_train_w_pdf(GM_EM1, GM_EM2, min(min(X1), min(X2)), max(max(X1), max(X2)))
#
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc='upper left')
plt.savefig(datetime.datetime.now().strftime('%d%m%Y_%H%M%S') + '.png')
plt.show()
