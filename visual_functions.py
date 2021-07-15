#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import tensorflow as tf


def plot_single_event(display, params, scaling=4):
    nofEvents = display.shape[0]
    display = cv2.resize(display, (display.shape[1]*scaling,
                                   display.shape[0]*scaling),
                         interpolation=cv2.INTER_AREA)

    # split list into chunks of three for each ring
    params = [params[i:i + 3] for i in range(0, len(params), 3)]

    # iterate over all rings
    for x, y, rad in params:
        r = int(rad*scaling)
        center_x = int(y * scaling)
        center_y = int(x * scaling)

        display = cv2.circle(display, (center_x, center_y), r, (1,1,1), 1)
    return display

def fit_rings(display, params, scaling=5):
    display = cv2.resize(display, (display.shape[1]*scaling,
                                   display.shape[0]*scaling))
#                         interpolation=cv2.INTER_AREA)

    # fit every ring in display
    for x, y, rad in params:
        r = int(rad*scaling)
        center_x = int((rad +.5 + y) * scaling)
        center_y = int((rad +.5 + x) * scaling)

        display = cv2.circle(display, (center_x, center_y), r, (1,1,1), 1)
    return display

def compare_true_and_predict(X_test, y_test, model, seed=42):
    plt.rcParams['figure.figsize'] = [30, 10]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    N = 8

    rand.seed(seed)
    indices = rand.sample(range(0, 500), N)

    fig, ax = plt.subplots(1,N)
    for n, m in zip(range(N), indices):
        img = plot_single_event(X_test[m], y_test[m])
        ax[n].imshow(img)

    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(1,N)
    for n, m in zip(range(N), indices):
        img = np.expand_dims(X_test[m], 0)
        plot = plot_single_event(X_test[m], y_pred[m])
        ax[n].imshow(plot)


def blend_plot(img1, img2, weights=(1, 1), show_plot=True):
    blend = cv2.addWeighted(img1, weights[0], img2, weights[1], 0.0)
    if show_plot:
        plt.imshow(blend, cmap="gray")
        plt.show()
    return blend
