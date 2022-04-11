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

def img_preprocessing(img):
    img = cv2.resize(img, (128,128))#, interpolation=cv2.INTER_AREA)
#    img = cv2.blur(img, (5,5))
    img = cv2.merge((img,img,img))
    return img

def plot_single_event(X, Y, scaling=5):
    X = cv2.resize(X, (X.shape[1]*scaling,
                       X.shape[0]*scaling),
                   interpolation=cv2.INTER_AREA)

    # split list into chunks of five for each ellipse
    Y = [Y[i:i + 5] for i in range(0, len(Y), 5)]

    # iterate over all rings
    for x, y, major, minor, angle in Y:
        center_x = int(y * scaling)# * X.shape[1])
        center_y = int(x * scaling)# * X.shape[0])
        major = int(major * scaling)# * X.shape[0])
        minor = int(minor * scaling)# * X.shape[1])

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

def show_predict(X_test, model, M, N, seed=0):
    plt.rcParams['figure.figsize'] = [30, 10]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

    rand.seed(seed)
    indices = rand.sample(range(0, X_test.shape[0]), M*N)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(M,N)
    for n, m in zip(product(range(M), range(N)), indices):
        plot = plot_single_event(X_test[m], y_pred[m])
        ax[n].imshow(plot)
#    fig.tight_layout()
#    fig.savefig("plots/{}x{}_predictions.png".format(M, N))

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
    # load data file and some preprocessing
    hits_true = loadDataFile("datasets/hits_true.txt")


    from create_data import *
    ins, os, hpr, rn = (72,32,1), 15, (24, 44), 0.08
    gen = SynthGen(ins, os, hpr, rn)
    X, y = gen.create_dataset(100)
    model = tf.keras.models.load_model("models/bmsf.model")
    show_predict(np.array(hits_true[:200]), model, 2, 3, 4)
    plt.show()


    #f = "datasets/mcbm.root"
    #imgs = np.array([get_data(f, i)[0] for i in range(100)])
    #display_data(imgs, 0)

    #x,y = get_data(f, 1)
    #img = plot_single_event(x, y)
    #plt.imshow(img)
    #plt.show()
