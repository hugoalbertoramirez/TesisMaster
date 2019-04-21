import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time


def build_dataset():
    file = 'C:\\Users\HugoAlberto\Desktop\Git\TesisMaster\CreditCardFraudDetection\creditcard.csv'
    data = pd.read_csv(file)

    # under-sample umbalanced set
    no_fraud, fraud = data['Class'].value_counts()
    index_no_fraud = np.random.choice(data[data['Class'] == 0].index, 10) #fraud
    index_fraud = np.array(data[data['Class'] == 1].index)
    index_fraud = np.random.choice(data[data['Class'] == 1].index, 10) #borrar linea
    index_total = np.concatenate((index_no_fraud, index_fraud))
    data = data.iloc[index_total]

    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    X_train = train_set.iloc[:, 0:-1]
    y_train = train_set['Class']
    y_train = (y_train == 1).astype(int)

    X_test = test_set.iloc[:, 0:-1]
    y_test = test_set['Class']
    y_test = (y_test == 1).astype(int)

    return X_train, y_train, X_test, y_test


def define_models(X_train, y_train, X_test, y_test, C_values, titles):
    plots = {}

    for i in range(len(titles)):
        p_train = []
        p_test = []
        r_test = []
        f1_test = []
        train_time = []

        for C in C_values:
            if i == 0:
                #model = svm.SVC(kernel='linear', C=C)
                model = svm.LinearSVC(C=C)
            elif i == 1:
                model = svm.LinearSVC(C=C)
            elif i == 3:
                model = svm.SVC(kernel='rbf', gamma=0.7, C=C)
            else:
                model = svm.SVC(kernel='poly', degree=3, C=C)

            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()

            y_pred_train = model.predict(X_train)
            prec_train = precision_score(y_train, y_pred_train, pos_label=0)

            y_pred_test = model.predict(X_test)
            prec_test = precision_score(y_test, y_pred_test, pos_label=0)

            recall_test = recall_score(y_test, y_pred_test, pos_label=0)

            f1_ = f1_score(y_test, y_pred_test, pos_label=0)

            p_train.append(prec_train)
            p_test.append(prec_test)
            r_test.append(recall_test)
            f1_test.append(f1_)
            train_time.append(end - start)

        plots[titles[i]] = {'train': p_train, 'test': p_test, 'recall': r_test, 'f1': f1_test, 'time': train_time}

    return plots


def plot_SVM(plots, C_values, titles):
    sns.set()
    fig, ax1 = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(len(plots)):
        plt.subplot(2, 2, i + 1)
        plt.title(titles[i])
        plt.xlabel('C')
        plt.ylabel('score')
        plt.ylim(0, 1.1)
        #plt.semilogx(C_values, plots[titles[i]]['train'], label='prec train')

        plt.semilogx(C_values, plots[titles[i]]['test'], label='prec')
        plt.semilogx(C_values, plots[titles[i]]['recall'], label='recall')
        plt.semilogx(C_values, plots[titles[i]]['f1'], label='f1')
        plt.legend(loc=6)

        ax2 = ax1[int(i / 2), i % 2].twinx()
        ax2.set_ylabel('segundos')
        #ax2.set_ylim(0, 0.011)
        ax2.grid(False)
        ax2.semilogx(C_values, plots[titles[i]]['time'], sns.xkcd_rgb["dark green"], label='train time')
        plt.legend(loc=4)

    plt.show()


C_values = np.logspace(-3, 2, 5)
titles = ['SVM con kernel lineal',
          'Linear SVM',
          'SVM con kernel rbf',
          'SVM con kernel polinominal grado 3'
          ]

X_train, y_train, X_test, y_test = build_dataset()
plots = define_models(X_train, y_train, X_test, y_test, C_values, titles)
plot_SVM(plots, C_values, titles)


