"""
__author__ = Nicolas Perez-Nieves
__email__ = nicolas.perez14@imperial.ac.uk

SDNN Implementation based on Kheradpisheh, S.R., et al. 'STDP-based spiking deep neural networks 
for object recognition'. arXiv:1611.01421v1 (Nov, 2016)
"""

from SDNN_cuda import SDNN
from Classifier import Classifier
from model_def import models
import numpy as np
from os.path import dirname, realpath
from math import floor
import os
import re
import sys

import time


def main(model_name, learn_SDNN=False, fine_tune=False):
    """
    :param model_name: model name to use
    :param learn_SDNN: if use stdp to train
    :param fine_tune:  if load pretrained weights
    :return:
    """
    if models[model_name] == None:
        print('No such model[%s] in model_def, exiting' % model_name)
        return
    # Flags
    # learn_SDNN = False  # This flag toggles between Learning STDP and classify features
                        # or just classify by loading pretrained weights for the face/motor dataset
    if learn_SDNN:
        if fine_tune:
            set_weights = True

        else:
            set_weights = False
        save_weights = True  # Saves the weights in a path (path_save_weigths)
        save_features = True  # Saves the features and labels in the specified path (path_features)
    else:
        set_weights = True  # Loads the weights from a path (path_set_weigths) and prevents any SDNN learning
        save_weights = False  # Saves the weights in a path (path_save_weigths)
        save_features = True  # Saves the features and labels in the specified path (path_features)

    # Results directories
    path_set_weigths = models[model_name]['output_path']
    path_save_weigths = models[model_name]['output_path']
    path_features = models[model_name]['output_path']

    # Create network
    first_net = SDNN(models[model_name]['network_params'], models[model_name]['weight_params'],
                     models[model_name]['stdp_params'], models[model_name]['total_time'],
                     DoG_params=models[model_name]['DoG_params'],
                     spike_times_learn=models[model_name]['learn_path'],
                     spike_times_train=models[model_name]['train_path'],
                     spike_times_test=models[model_name]['test_path'], device='GPU')

    # Set the weights or learn STDP
    if set_weights:
        network_layer_num = len(models[model_name]['network_params'])
        weight_path_list = [path_set_weigths + 'weight_' + str(i) + '.npy' for i in range(network_layer_num - 1)]
        first_net.set_weights(weight_path_list)
    if learn_SDNN:
        first_net.train_SDNN()

    # Save the weights
    if save_weights:
        weights = first_net.get_weights()
        for i in range(len(weights)):
            np.save(path_save_weigths + 'weight_'+str(i), weights[i])

    # Get features
    X_train, y_train = first_net.train_features()
    X_test, y_test = first_net.test_features()

    # Save X_train and X_test
    if save_features:
        np.save(path_features + 'X_train', X_train)
        np.save(path_features + 'y_train', y_train)
        np.save(path_features + 'X_test', X_test)
        np.save(path_features + 'y_test', y_test)

    # ------------------------------- Classify -------------------------------#
    classifier_params = {'C': 1.2, 'gamma': 'auto'}
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    X_train -= train_mean
    X_test -= train_mean
    X_train /= (train_std + 1e-5)
    X_test /= (train_std + 1e-5)
    svm = Classifier(X_train, y_train, X_test, y_test, classifier_params, classifier_type='SVM')
    train_score, test_score = svm.run_classiffier()
    print('Train Score: ' + str(train_score))
    print('Test Score: ' + str(test_score))

    print('DONE')


if __name__ == '__main__':
    start = time.time()
    model_name = 'default'
    isLearning = False
    fine_tune = False
    if len(sys.argv) > 2:
        model_name = sys.argv[1]
        if sys.argv[2] == 'train':
            isLearning = True
        elif sys.argv[2] == 'finetune':
            isLearning = True
            fine_tune = True
    main(model_name, isLearning, fine_tune)
    end = time.time()
    print('Time Userd', end-start)
