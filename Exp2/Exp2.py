import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
sys.path.append('C:\\Users\\HugoAlberto\Desktop\Git\Tesis\Exp2')
from GM1D import GM1D
from SVM1D import SVM1D

GM1 = GM1D('class 1', color='r', form='s')
GM2 = GM1D('class 2', color='b', form='*')

#GM1.create_random(0, 500, 2)
#GM2.create_random(500, 1000, 2)

GM1.create_from_data(np.array([[100, 10], [300, 10]], dtype=np.float32), [.3, .7])
GM2.create_from_data(np.array([[700, 200], [900, 10]], dtype=np.float32), [.9, .1])

plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

X1, GM_EM1 = GM1.run(1000)
X2, GM_EM2 = GM2.run(1000)

svm = SVM1D()

svm.train_w_all_points(X1, X2)
svm.train_w_means_pi(GM_EM1, GM_EM2)
svm.svm_train_w_pdf(GM_EM1, GM_EM2, min(min(X1), min(X2)), max(max(X1), max(X2)))

plt.xlabel("x")
plt.ylabel("MG(x)")
plt.legend(loc='upper left')
plt.savefig(datetime.datetime.now().strftime('%d%m%Y_%H%M%S') + '.png')
plt.show()
