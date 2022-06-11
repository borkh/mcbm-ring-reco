#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import tensorflow as tf
from tqdm import tqdm
from itertools import product
import ROOT
import pandas as pd

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

def plot_single_event(X, Y, scaling=5):
    X = cv2.resize(X, (X.shape[1]*scaling,
                       X.shape[0]*scaling),
                   interpolation=cv2.INTER_AREA)

    # split list into chunks of five for each ellipse
    Y = np.array([Y[i:i + 5]*scaling for i in range(0, len(Y), 5)])
    # iterate over all rings
    for center_x, center_y, major, minor, angle in Y.astype(int):
        X = cv2.ellipse(X, (center_x, center_y), (major, minor),
                        angle + 90, 0, 360, (1,1,1))

    return X

def compare_true_and_predict(X_test, y_test, model, seed=42, show_true=True):
    plt.rcParams['figure.figsize'] = [30, 10]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    N = 8

    rand.seed(seed)
    indices = rand.sample(range(0, y_test.shape[0]), N)

    if show_true:
        fig, ax = plt.subplots(1,N)
        for n, m in zip(range(N), indices):
            img = plot_single_event(X_test[m], y_test[m])
            ax[n].imshow(img)

    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(1,N)
    for n, m in zip(range(N), indices):
        plot = plot_single_event(X_test[m], y_pred[m])
        ax[n].imshow(plot)

def show_predict(X_test, model, seed=0):
    y_pred = model.predict(X_test)
    displays = np.array([plot_single_event(X_test[i], y_pred[i]) for  i in range(y_pred.shape[0])])
    display_data(displays, seed)

def display_data(imgs, seed=42):
    plt.rcParams['figure.figsize'] = [30, 10]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    N = 8

    rand.seed(seed)
    indices = rand.sample(range(0, imgs.shape[0]), N)

    fig, ax = plt.subplots(1,N)
    for n, m in zip(range(N), indices):
        ax[n].imshow(imgs[m])
    plt.show()

def loadDataFile(datafile, pixel_x = 32, pixel_y = 72):
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

if __name__ == '__main__':
    path = "data/runx.rec.root"
    plot_root(path)


"""
    from tensorflow_addons.optimizers import *
    import matplotlib
    font = {'family' : 'normal',
            'size'   : 50}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['lines.linewidth'] = 4
    matplotlib.rcParams['axes.linewidth'] = 0

    epochs = 24
    spe = 5000
    ep = np.arange(0, epochs)
    lr = []
    m = Triangular2CyclicalLearningRate(1e-6, 0.05, 4*spe)

    for e in range(epochs):
        steps = np.arange(e*spe, (e+1)*spe)
        lr.append(np.mean(m(steps)))
    steps = np.arange(0, epochs*spe)
    plt.plot(steps/spe, m(steps))
    plt.ylabel("learning rate")
    plt.xlabel("epoch")
    plt.subplots_adjust(left=0.11, bottom=0.13, right=0.98, top=0.98)
    plt.grid()
    plt.show()
"""
"""
    hits_true = loadDataFile("datasets/hits_true.txt")

    from create_data import *
    ins, os, hpr, rn = (72,32,1), 15, (24, 44), 0.08
    gen = SynthGen(ins, os, hpr, rn)
    X, y = gen.create_dataset(100)
    model = tf.keras.models.load_model("models/bmsf.model")
    show_predict(np.array(hits_true[:200]), model, 2, 3, 4)
    plt.show()
"""
