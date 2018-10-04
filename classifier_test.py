from sklearn import datasets
from Classifier import Classifier
from matplotlib import pyplot as plt
import numpy as np
import os


def load_features(dir_path):
    file_names = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    ret = [None, None, None, None]
    for i, file_name in enumerate(file_names):
        path = os.path.join(dir_path, file_name)
        if os.path.exists(path):
            ret[i] = np.load(path)
    return ret


if __name__ == "__main__":
    # iris = datasets.load_iris()
    features = load_features('./results')
    svm = Classifier(features[0], features[1], features[2], features[3], {'C':1.0})
    svm.cross_val_svm({'C':np.arange(1,101).tolist()}, 3)
    print("Best Score: ", max(svm.cvs_mean), max(svm.cvs_std))
    plt.show(svm.plots[-1])