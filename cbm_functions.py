#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from sklearn.datasets import make_circles
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer, Conv2D, BatchNormalization,
                                     Dropout, MaxPooling2D, Flatten, Dense)


# ------------------------------------------------------------------------------------
# ------ creating datasets -----------------------------------------------------------

def create_event(nofRings, display_size=32, limits=(-7, 23, -7, 23)):
    minX, maxX, minY, maxY = limits
    # create empty display
    display = np.zeros((display_size, display_size, 1))
    params = np.zeros(6)
    pars = []

    for _ in range(nofRings):
        X, y = make_circles(noise=.05, factor=.1, n_samples=(30,0))
        radius = rand.uniform(12, 18)/2

        #increase raidus of ring and move it into origin of display
        X = np.round(X*radius + radius, 0).astype('int32')

        # move center of ring to random location inside display
        xshift = int(rand.uniform(minX, maxX))
        yshift = int(rand.uniform(minY, maxY))
        X[:,0] += xshift
        X[:,1] += yshift

        # set the values of the positions of the circles in the display image to 1
        for x, y in zip(X[:,0], X[:,1]):
            if x > 0 and x < display_size and y > 0 and y < display_size:
                display[x,y] = 1

        # set the values of the center of the circles in the feature image to 1
        center_x, center_y, radius = xshift + radius + 0.5, yshift + radius + 0.5, radius + 0.33
        pars.extend([center_x, center_y, radius])

    for n in range(len(pars)):
        params[n] = pars[n]

    return display, params


def create_dataset(nofEvents):
    displays, pars = [], []
    for _ in tqdm(range(nofEvents)):
        display, params = create_event(int(rand.uniform(1, 2)))
        displays.append(display)
        pars.append(params)
    return np.array(displays), np.array(pars)


# -----------------------------------------------------------------------------------
# ------ model functions ------------------------------------------------------------

def create_model(input_shape, nofLayers, layer_size=16,  activation='relu', initializer='glorot_normal'):
    model = Sequential()
    model.add(InputLayer(input_shape))

    for n in range(nofLayers):
        model.add(Conv2D(layer_size * 2**n, (3,3), padding='same', activation=activation,
                         kernel_initializer=initializer,
                         kernel_regularizer=None))
        model.add(BatchNormalization(momentum=0.99))
#        model.add(Dropout(0.2))

    model.add(Conv2D(1, (3,3), padding='same', activation=activation,
                     kernel_initializer=initializer,
                     kernel_regularizer=None))
    return model

def custom_loss_function(y_true, y_pred):
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)

# ------------------------------------------------------------------------------------
# ------ visual functions ------------------------------------------------------------

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

if __name__ == "__main__":
    print(tf.config.list_physical_devices())
    # training data
    displays, params = create_dataset(20000)

    data_dir = "./datasets/"
    np.save(data_dir + "displays.npy", displays)
    np.save(data_dir + "params.npy", params)

    # testing data
    displays, params = create_dataset(1000)

    data_dir = "./datasets/"
    np.save(data_dir + "displays_test.npy", displays)
    np.save(data_dir + "params_test.npy", params)
