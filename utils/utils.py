#!/usr/bin/env python3
import cv2, numpy as np, pandas as pd, pickle as pkl
import tensorflow as tf
from itertools import product
from functools import reduce

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# %%
# functions for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_single_event(X, Y1=None, Y2=None, Y3=None, Y4=None,  scaling=10):
    X = cv2.resize(X, (X.shape[1]*scaling,
                       X.shape[0]*scaling),
                   interpolation=cv2.INTER_AREA)

    # iterate over all rings
    if Y1 is not None:
        for ring in (Y1*scaling).astype(int):
            try:
                X = cv2.ellipse(X, (ring[0], ring[1]),
                                   (ring[2], ring[3]),
                                   ring[4] + 90, 0, 360,
                                   (0,1,0), 2)
            except cv2.error as e:
                print(e)

    if Y2 is not None:
        for ring in (Y2*scaling).astype(int):
            try:
                X = cv2.ellipse(X, (ring[0], ring[1]),
                                   (ring[2], ring[3]),
                                   ring[4] + 90, 0, 360,
                                   (1,0,0), 2)
            except cv2.error as e:
                print(e)

    if Y3 is not None:
        for ring in (Y3*scaling).astype(int):
            try:
                X = cv2.ellipse(X, (ring[0], ring[1]),
                                   (ring[2], ring[3]),
                                   ring[4] + 90, 0, 360,
                                   (1,1,0), 2)
            except cv2.error as e:
                print(e)

    if Y4 is not None:
        for ring in (Y4*scaling).astype(int):
            try:
                X = cv2.ellipse(X, (ring[0], ring[1]),
                                   (ring[2], ring[3]),
                                   ring[4] + 90, 0, 360,
                                   (0,1,1), 2)
            except cv2.error as e:
                print(e)
    return X

def display_images(M, N, images, n_plots, scaling=1):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    yticks = np.array([0,10,20,30,40,50,60,70])
    xticks = np.array([0,10,20,30])
    for n in range(n_plots):
        fig, ax = plt.subplots(M,N)
        for i, img in zip(product(range(M),range(N)), images[n*M*N:n*M*N + M*N]):
        #for i, img in enumerate(images[n*M*N:n*M*N + M*N]):
            ax[i].imshow(img)
            ax[i].set_yticks(yticks*scaling, yticks)
            ax[i].set_xticks(xticks*scaling, xticks)
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
            try:
                params[i,j] = np.round(par, 2)
            except IndexError as e:
                print(e)
    return params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def filter_events(y):
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

    indices6 = np.where(y[:,0] <= 72.)[0]
    indices7 = np.where(y[:,5] <= 72.)[0]
    indices8 = np.where(y[:,10] <= 72.)[0]
    indices9 = np.where(y[:,15] <= 72.)[0]
    indices10 = np.where(y[:,20] <= 72.)[0]

    indices11 = np.where(y[:,1] >= 0.)[0]
    indices12 = np.where(y[:,6] >= 0.)[0]
    indices13 = np.where(y[:,11] >= 0.)[0]
    indices14 = np.where(y[:,16] >= 0.)[0]
    indices15 = np.where(y[:,21] >= 0.)[0]

    indices16 = np.where(y[:,1] <= 72.)[0]
    indices17 = np.where(y[:,6] <= 72.)[0]
    indices18 = np.where(y[:,11] <= 72.)[0]
    indices19 = np.where(y[:,16] <= 72.)[0]
    indices20 = np.where(y[:,21] <= 72.)[0]

    indices21 = np.where(y[:,2] <= 20.)[0]
    indices22 = np.where(y[:,7] <= 20.)[0]
    indices23 = np.where(y[:,12] <= 20.)[0]
    indices24 = np.where(y[:,17] <= 20.)[0]
    indices25 = np.where(y[:,22] <= 20.)[0]

    indices = reduce(np.intersect1d, (indices1, indices2, indices3, indices4, indices5,
                                      indices6, indices7, indices8, indices9, indices10,
                                      indices11, indices12, indices13, indices14, indices15,
                                      indices16, indices17, indices18, indices19, indices20,
                                      indices21, indices22, indices23, indices24, indices25)) # combine all filters

    return indices # return filtered images with corresponding parameters
