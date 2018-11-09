"""
This is the file which defines the models that
will be used in the main_universal.py

Each model contains:
    network parameter
    weight initialize parameter
    stdp parameter
    simulation time
    DoG parameter
    train test data paths
    output weights paths if needed
"""
import math
import numpy as np

models = {}
# ---------------------------------------------------default start
model_name = 'default'
DoG_params = {'img_size': (256, 160), 'DoG_size': 7, 'std1': 1., 'std2': 2.}  # img_size is (col size, row size)
total_time = 15
network_params = [{'Type': 'input', 'num_filters': 1, 'pad': (0, 0), 'H_layer': DoG_params['img_size'][1],
                       'W_layer': DoG_params['img_size'][0]},
                  {'Type': 'conv', 'num_filters': 4, 'filter_size': 5, 'th': 10.},
                  {'Type': 'pool', 'num_filters': 4, 'filter_size': 7, 'th': 0., 'stride': 6},
                  {'Type': 'conv', 'num_filters': 20, 'filter_size': 17, 'th': 60.},
                  {'Type': 'pool', 'num_filters': 20, 'filter_size': 5, 'th': 0., 'stride': 5},
                  {'Type': 'conv', 'num_filters': 20, 'filter_size': 5, 'th': 2.}]
weight_params = {'mean': 0.8, 'std': 0.01}
# max_learn_iter = [0, 3000, 0, 5000, 0, 5000]
max_learn_iter = [0, 0, 0, 0, 0, 5000]
stdp_params = {'max_learn_iter': max_learn_iter,
                   'stdp_per_layer': [0, 10, 0, 5, 0, 1],
                   'max_iter': sum(max_learn_iter),
                   'a_minus': np.array([0, .003, 0, .0003, 0, .0003], dtype=np.float32),
                   'a_plus': np.array([0, .004, 0, .0004, 0, .0004], dtype=np.float32),
                   'offset_STDP': [0,
                                   math.floor(network_params[1]['filter_size']),
                                   0,
                                   math.floor(network_params[3]['filter_size']/8),
                                   0,
                                   math.floor(network_params[5]['filter_size'])]
                   }
path = '.'
spike_times_learn = [path + '/datasets/LearningSet/Face/', path + '/datasets/LearningSet/Motor/']
spike_times_train = [path + '/datasets/TrainingSet/Face/', path + '/datasets/TrainingSet/Motor/']
spike_times_test = [path + '/datasets/TestingSet/Face/', path + '/datasets/TestingSet/Motor/']
output_path = 'results/'
models.update({model_name:{'DoG_params':DoG_params,
                           'total_time':total_time,
                           'network_params':network_params,
                           'weight_params':weight_params,
                           'stdp_params':stdp_params,
                           'learn_path':spike_times_learn,
                           'train_path':spike_times_train,
                           'test_path':spike_times_test,
                           'output_path':output_path}})
# ---------------------------------------------------default end-------------------------------------------------------
