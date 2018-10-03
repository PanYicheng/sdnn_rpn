from sklearn import datasets
from Classifier import Classifier
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    iris = datasets.load_iris()
    svm = Classifier(iris.data, iris.target,iris.data, iris.target, {'C':1.0})
    svm.cross_val_svm({'C':np.arange(1,101).tolist()}, 3)
    # print(svm.cvs_mean, svm.cvs_std)
    plt.show(svm.plots[-1])