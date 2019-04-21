import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import time

def build_dataset():
    file = 'C:\\Users\HugoAlberto\Desktop\Git\TesisMaster\MNIST_SkinCancer\hmnist_8_8_L.csv'
    data = pd.read_csv(file)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    class_label = 4

    X_train = train_set.iloc[:, 0:-1]
    y_train = train_set['label']
    y_train = (y_train == class_label).astype(int)

    X_test = test_set.iloc[:, 0:-1]
    y_test = test_set['label']
    y_test = (y_test == class_label).astype(int)

    return X_train, y_train, X_test, y_test


def build_dataset_stratified():
    file = 'C:\\Users\HugoAlberto\Desktop\Git\TesisMaster\MNIST_SkinCancer\hmnist_8_8_L.csv'
    data = pd.read_csv(file)

    class_label = 4

    strat_train_set = []
    strat_test_set = []
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42) # 0.23574638-> N = 2361
    for train_index, test_index in split.split(data, data['label']):
        strat_test_set = data.loc[train_index]
        strat_train_set = data.loc[test_index]


    X_train = strat_train_set.iloc[:, 0:-1]
    y_train = strat_train_set['label']
    y_train = (y_train == class_label).astype(int)

    X_test = strat_test_set.iloc[:, 0:-1]
    y_test = strat_test_set['label']
    y_test = (y_test == class_label).astype(int)

    return X_train, y_train, X_test, y_test


def plot_hist():
    sns.set()
    file = 'C:\\Users\HugoAlberto\Desktop\Git\TesisMaster\MNIST_SkinCancer\hmnist_8_8_L.csv'
    data = pd.read_csv(file)

    class_label = 4

    strat_train_set = []
    strat_test_set = []
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.23574638, random_state=42) # 0.23574638-> N = 2361
    for train_index, test_index in split.split(data, data['label']):
        strat_test_set = data.loc[train_index]
        strat_train_set = data.loc[test_index]

    labels = ['Enfermedad de\nBowen',
              'Cárcinoma de\ncélulas basales',
              'Lesiones benignas\nsimilares a la queratosis',
              'Dermatofibroma',
              'Melanoma',
              'Nevos melanocíticos',
              'Lesiones vasculares\ny Granulomas piógenos']

    ax = sns.countplot(y="label", color='c', data=strat_train_set)
    plt.xlabel('Frecuencia')
    plt.ylabel('Clase')
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.show()


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
                model = svm.SVC(kernel='linear', C=C, verbose=1)
            elif i == 1:
                model = svm.LinearSVC(C=C, verbose=1)
            elif i == 3:
                model = svm.SVC(kernel='rbf', gamma=0.7, C=C, verbose=1)
            else:
                model = svm.SVC(kernel='poly', degree=3, C=C, verbose=1)

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


def plot_sep_SVM(plots, C_values, titles):
    sns.set()
    #fig, ax1 = plt.subplots()
    #fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(len(plots)):
        fig, ax1 = plt.subplots()
        #plt.subplot(2, 2, i + 1)
        plt.title(titles[i])
        plt.xlabel('C')
        plt.ylabel('score')
        plt.ylim(0, 1.1)
        #plt.semilogx(C_values, plots[titles[i]]['train'], label='prec train')

        plt.semilogx(C_values, plots[titles[i]]['test'], label='prec')
        plt.semilogx(C_values, plots[titles[i]]['recall'], label='recall')
        plt.semilogx(C_values, plots[titles[i]]['f1'], label='f1')
        plt.legend(loc=6)

        ax2 = ax1.twinx()
        ax2.set_ylabel('segundos')
        ax2.set_ylim(0, 0.011)
        ax2.grid(False)
        ax2.semilogx(C_values, plots[titles[i]]['time'], sns.xkcd_rgb["dark green"], label='train time')
        plt.legend(loc=4)
        plt.show()

    #plt.show()

C_values = np.logspace(-3, 2, 5)
titles = ['SVM con kernel lineal',
          'Linear SVM',
          'SVM con kernel rbf',
          'SVM con kernel polinominal grado 3'
          ]

X_train, y_train, X_test, y_test = build_dataset_stratified()
plots = define_models(X_train, y_train, X_test, y_test, C_values, titles)
plot_SVM(plots, C_values, titles)


