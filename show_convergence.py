"""
    This file is used to show the convergence during training process
    Usage:
        python show_convergence.py converge_file
    Note:
        converge_file must be the file writed by the training process in SDNN_cuda.py
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

if __name__ == "__main__":
    if len(sys.argv) > 1:
        weights_converg = np.load(sys.argv[1])
        layer_num = weights_converg.shape[1]
        cmap = get_cmap(layer_num)
        x = np.arange(0, weights_converg.shape[0])
        for i in range(0, layer_num, 2):
            conv = weights_converg[:,i]
            plt.plot(x, conv, colors=cmap(i), label="Conv Layer %d" % i)
        plt.legend()
        plt.show()
    else:
        print("Usage: python show_convergence.py converge_file_name")
