#!/usr/bin/env python3
import cv2, numpy as np, matplotlib.pyplot as plt, tensorflow as tf, pandas as pd
from itertools import product
from functools import reduce
from PIL import Image
import ROOT

import matplotlib
matplotlib.use('TkAgg')

def get_data(rootfile, index):
    f = ROOT.TFile.Open(rootfile)
    tree = f.Get("train")

    tree.GetEntry(index)
    return np.array([*tree.x]).reshape(72,32,1), np.array([*tree.y])

def plot_root(path):
    x = np.array([get_data(path, i)[0] for i in range(100)])
    y = np.array([get_data(path, i)[1] for i in range(100)])
    for i in range(3):
        plt.imshow(plot_single_event(x[i], y[i]))
        plt.show()
    #display_data(x, 0)

# %%
# functions for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_single_event(X, Y, scaling=10):
    X = cv2.resize(X, (X.shape[1]*scaling,
                       X.shape[0]*scaling),
                   interpolation=cv2.INTER_AREA)

    # split list into chunks of five for each ellipse
    Y = np.array([Y[i:i + 5]*scaling for i in range(0, len(Y), 5)], dtype=int)
    # iterate over all rings
    for center_x, center_y, major, minor, angle in Y.astype(int):
        X = cv2.ellipse(X, (int(center_x), int(center_y)), (int(major), int(minor)),
                        int(angle) + 90, 0, 360, (1,1,1), 2)

    return X

def display_images(M, N, images, n_plots):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    for n in range(n_plots):
        fig, ax = plt.subplots(M,N)
        #for i, img in zip(product(range(M),range(N)), images[n*M*N:n*M*N + M*N]):
        for i, img in enumerate(images[n*M*N:n*M*N + M*N]):
            ax[i].imshow(img)
        plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# functions for reading .csv file from simulation data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def loadFeatures(datafile, pixel_x = 32, pixel_y = 72):
    with open(datafile, 'r') as temp_f:
        col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
    column_names = [i for i in range(0, max(col_count))]
    hits = pd.read_csv(datafile,header=None ,index_col=0,comment='#',
                       delimiter=",", nrows=
                       20000,names=column_names).values.astype('int32')
    hits[hits < 0] = 0
    hits_temp = np.zeros([len(hits[:,0]), pixel_x*pixel_y])
    for i in range(len(hits[:,0])):
        for j in range(len(hits[0,:])):
            if hits[i,j]==0:
                break
            hits_temp[i,hits[i,j]-1]+=1
    hits_temp = tf.reshape(hits_temp, [len(hits[:,0]), pixel_y, pixel_x])
    hits_temp = tf.clip_by_value(hits_temp, clip_value_min=0., clip_value_max=1.)
    hits = tf.cast(hits_temp[..., tf.newaxis],dtype=tf.float32)
    print('load data from  ' + datafile + '  -> ' + str(len(hits[:])) + '  events loaded' )
    return hits

def loadParameters(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
        n = len(lines)
    params = np.zeros((n,25))
    for i, line in enumerate(lines):
        line = line.strip().split(",")
        line.remove("")
        line = np.array([float(x) for x in line])
        for j, par in enumerate(line):
            params[i,j] = np.round(par, 2)
    return params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def filter_events(x, y):
    cond1 = np.all(~np.isnan(y), axis=1) # remove events with NaN in parameters
    cond2 = np.invert(np.all(y == 0., axis=1)) # remove events with only zeros as parameters

    indices1 = np.where(cond1 & cond2)[0] # apply first two conditions
    # some events have coordinates at (-1.5,-1.5) that don't match rings correctly
    # remove those events for up to five rings
    indices2 = np.where(y[:,0] >= 0.)[0]
    indices3 = np.where(y[:,5] >= 0.)[0]
    indices4 = np.where(y[:,10] >= 0.)[0]
    indices5 = np.where(y[:,15] >= 0.)[0]
    indices5 = np.where(y[:,20] >= 0.)[0]

    indices = reduce(np.intersect1d, (indices1, indices2, indices3, indices4)) # combine all filters

    return x[indices], y[indices] # return filtered images with corresponding parameters
