#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import tensorflow as tf
from tqdm import tqdm
from itertools import product


def plot_single_event(display, params, scaling=4):
    display = cv2.resize(display, (display.shape[0]*scaling,
                                   display.shape[1]*scaling),
                         interpolation=cv2.INTER_AREA)

    # split list into chunks of five for each ellipse
    params = [params[i:i + 5] for i in range(0, len(params), 5)]

    # iterate over all rings
    for x, y, major, minor, angle in params:
        center_x = int(y * scaling)
        center_y = int(x * scaling)
        major = int(major * scaling)
        minor = int(minor * scaling)

        display = cv2.ellipse(display, (center_x, center_y), (major, minor),
                              angle+90, 0, 360, (1,1,1))
    return display

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

def show_predict(X_test, y_test, model, shift=0, save=None):
    plt.rcParams['figure.figsize'] = [30, 30]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    save_dir = "./pred_imgs/"
    N = 8

    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(N,N)
    for i, (n, m) in enumerate(product(range(N), range(N))):
        plot = plot_single_event(X_test[i+shift*64], y_pred[i+shift*64])
        ax[n][m].imshow(plot)

    if save is not None:
        plt.savefig(save_dir + save)

def save_pred_imgs(X_test, y_test, model):
    save_dir = "./pred_imgs/"
    y_pred = model.predict(X_test)
    print("Saving plots...")
    for n in tqdm(range(y_test.shape[0])):
        plot = plot_single_event(X_test[n], y_pred[n])
        plt.imshow(plot)
        plt.savefig(save_dir + "pred{}.png".format(n))
