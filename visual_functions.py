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

def get_data(rootfile, index):
    f = ROOT.TFile.Open(rootfile)
    tree = f.Get("train")

    tree.GetEntry(index)
    return np.array([*tree.x]).reshape(72,32,1), np.array([*tree.y])

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

def show_predict(X_test, model, seed=0):
    plt.rcParams['figure.figsize'] = [30, 10]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    N = 8

    rand.seed(seed)
    indices = rand.sample(range(0, X_test.shape[0]), N)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(1,N)
    for n, m in zip(range(N), indices):
        plot = plot_single_event(X_test[m], y_pred[m])
        ax[n].imshow(plot)

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

if __name__ == '__main__':
    train_dir = "./datasets/train/"
    test_dir = "./datasets/test/"
    for i in range(4):
        img = np.load(train_dir + "X/X{}.npz".format(0), "r")['arr_0'][i]
        params = np.load(train_dir + "y/y{}.npz".format(0), "r")['arr_0'][i]
        img = plot_single_event(img, params)
        print(params)
#
        plt.imshow(img)
        plt.show()

    #f = "datasets/mcbm.root"
    #imgs = np.array([get_data(f, i)[0] for i in range(100)])
    #display_data(imgs, 0)

    #x,y = get_data(f, 1)
    #img = plot_single_event(x, y)
    #plt.imshow(img)
    #plt.show()
