#!/usr/bin/env python3
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np, pandas as pd, tensorflow as tf, math
from model import *
from utils import *
from sweep_configs import *
#from train import *

sim_x = np.array(loadFeatures("../data/features.csv"))
ideal_hough_y = loadParameters("../data/targets_ring.csv")

sim_x, ideal_hough_y = filter_events(sim_x, ideal_hough_y) # filter events with incorrectly fitted rings

ideal_hough_y = ideal_hough_y.reshape(ideal_hough_y.shape[0], 5, 5)


if __name__ == '__main__':
    hough = np.array([plot_single_event(sim_x[i], ideal_hough_y[i]) for  i in range(sim_x.shape[0])])
    display_images(1,6, sim_x, 1)
    display_images(1,6, hough, 1)
