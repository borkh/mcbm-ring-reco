#!/usr/bin/env python3
import cv2, numpy as np, pandas as pd, pickle as pkl
import tensorflow as tf
from itertools import product
from functools import reduce
from PIL import Image
import ROOT

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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
def plot_single_event(X, Y1=None, Y2=None,  scaling=10):
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
                                   (1,1,0), 2)
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

if __name__ == '__main__':
    font = {'size'   : 26}
    matplotlib.rc('font', **font)
    plt.rc('lines', linewidth = 4)

    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()

    with open('data/sim+idealhough+hough+cnn.pkl', 'rb') as f:
        sim, idealhough, hough, cnn = pkl.load(f)
    sim = np.array([cv2.merge((a,a,a)) for a in sim])

    # calulate errors
    #print(mse(cnn, idealhough).numpy())
    #print(mse(hough, idealhough).numpy())

    print(f'''\n\nMSE\'s of cnn and idealhough: \t x: {mse(cnn[:,:,0],idealhough[:,:,0]).numpy()}
                                 y: {mse(cnn[:,:,1],idealhough[:,:,1]).numpy()}
                                r1: {mse(cnn[:,:,2],idealhough[:,:,2]).numpy()}
                                r2: {mse(cnn[:,:,3],idealhough[:,:,3]).numpy()}''')

    print(f'''\n\nMSE\'s of idealhough and hough: \t x: {mse(idealhough[:,:,0],hough[:,:,0]).numpy()}
                                 y: {mse(idealhough[:,:,1],hough[:,:,1]).numpy()}
                                r1: {mse(idealhough[:,:,2],hough[:,:,2]).numpy()}
                                r2: {mse(idealhough[:,:,3],hough[:,:,3]).numpy()}''')

    print(f'''\n\nMSE\'s of cnn and hough: \t x: {mse(cnn[:,:,0],hough[:,:,0]).numpy()}
                                 y: {mse(cnn[:,:,1],hough[:,:,1]).numpy()}
                                r1: {mse(cnn[:,:,2],hough[:,:,2]).numpy()}
                                r2: {mse(cnn[:,:,3],hough[:,:,3]).numpy()}''')



    # %%
    # create ring fits
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #idealhough_fit = np.array([plot_single_event(x, y1) for x,y1 in zip(sim[:50], idealhough[:50])])
    """
    cnn_fit = np.array([plot_single_event(x, y1) for x,y1 in zip(sim[500:1000], cnn[500:1000])])

    cnn_vs_idealhough = np.array([plot_single_event(x,y1,y2) for x,y1,y2 in zip(sim[500:1000], cnn[500:1000], idealhough[500:1000])])
    #cnn_vs_hough = np.array([plot_single_event(x,y1,y2) for x,y1,y2 in zip(sim[:500], cnn[:500], hough[:500])])
    #hough_vs_idealhough = np.array([plot_single_event(x,y1,y2) for x,y1,y2 in zip(sim[:50], hough[:50], idealhough[:50])])

    #display_images(2,5,cnn_fit,5,10)
    #display_images(4,5,cnn_vs_idealhough,20,10)
    #display_images(3,5,cnn_vs_hough,5,10)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # %%
    # plot bad ring fits
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bad_cnn_fits = []
    bad_cnn_idx = []
    y1, y2 = cnn, idealhough
    for i in range(500):
        err = mse(y1[i,:,0],y2[i,:,0]).numpy()
        if err > 30.:
            print(y1[i,:,0], y2[i,:,0])
            print(mse(y1[i],y2[i]).numpy())
            bad_cnn_fits.append(cnn_vs_idealhough[i])
            bad_cnn_idx.append(i)
    display_images(2,5,bad_cnn_fits,5,10)
    """

    # %%
    # plot histograms
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_rings_cnn = []
    for y in cnn:
        n = len(np.where(np.invert(np.all(y == 0., axis=1)))[0])
        if n <= 4:
            n_rings_cnn.append(n)

    n_rings_idealhough = []
    for y in idealhough:
        n = len(np.where(np.invert(np.all(y == 0., axis=1)))[0])
        if n <= 4:
            n_rings_idealhough.append(n)

    n_rings_hough = []
    for y in hough:
        n = len(np.where(np.invert(np.all(y == 0., axis=1)))[0])
        if n <= 4:
            n_rings_hough.append(n)

    plt.hist(n_rings_cnn, bins=range(6), align='left', histtype='step', linewidth=6)
    plt.hist(n_rings_idealhough, bins=range(6), align='left', histtype='step', linewidth=6)
    plt.hist(n_rings_hough, bins=range(6), align='left', histtype='step', linewidth=6)
    plt.ylabel('count')
    plt.xlabel('number of rings per event')
    plt.show()
    """
    fig, ax = plt.subplots(1,3)
    ax[0].hist(n_rings_cnn, bins=4)
    ax[0].set_ylabel('number of rings per event')
    ax[0].set_title('regression CNN')

    ax[1].hist(n_rings_idealhough, bins=4)
    ax[1].set_ylabel('number of rings per event')
    ax[1].set_title('ideal HTM')

    ax[2].hist(n_rings_hough, bins=4)
    ax[2].set_ylabel('number of rings per event')
    ax[2].set_title('HTM')
    plt.show()
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
