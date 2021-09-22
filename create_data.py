#!/usr/bin/env python3
import numpy as np
import random as rand
from sklearn.datasets import make_circles
from tqdm import tqdm
import os

def rotate(img, angle):
    # transorm angle from deg to rad
    angle *= np.pi / 180
    rotation_matrix = np.array([np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]).reshape(2,2)
    return img @ rotation_matrix

def create_event(nofRings, display_size=48, limits=(7, 41, 7, 41)):
    minX, maxX, minY, maxY = limits
    # create empty display
    display = np.zeros((display_size, display_size, 1))
    params = np.zeros(15)
    pars = []

    # create noise
    noise = rand.randint(4, 7)
    for _ in range(noise):
        x, y  = rand.randint(minX, maxX), rand.randint(minY, maxY)
        display[x,y] = 1

    # create ellipses
    for _ in range(nofRings):
        X, y = make_circles(noise=.09, factor=.1, n_samples=(rand.randint(15, 30),0))

        # define semi-major and semi-minor axes of ellipse
        r = rand.randint(6,8)

        ## uncomment to create ellipses
        #major, minor = r + rand.randint(1, 2), r + rand.randint(1, 2)

        # uncomment to create rings
        major, minor = r, r

        X[:,0] *= major
        X[:,1] *= minor

        # rotate ellipse
        if major == minor:
            angle = 0 # angles of rings are always set to zero
        else:
            angle = rand.randint(0, 90)
        X = rotate(X, angle)

        # convert all entries to integers
        X = np.round(X, 0).astype('int32')

        # move center of ellipse to random location inside display
        xshift, yshift = rand.randint(minX, maxX), rand.randint(minY, maxY)
        X[:,0] += xshift
        X[:,1] += yshift

        # set the values of the positions of the circles in the display image to 1
        for x, y in zip(X[:,0], X[:,1]):
            if x > 0 and x < display_size and y > 0 and y < display_size:
                display[x,y] = 1

        center_x, center_y = xshift + 0.5, yshift + 0.5
        pars.extend([center_x, center_y, major, minor, angle])

    for n in range(len(pars)):
        params[n] = pars[n]

    return display, params

def create_dataset(nofEvents):
    displays, pars = [], []
    for _ in tqdm(range(nofEvents)):
        display, params = create_event(rand.randint(1, 3))
        displays.append(display)
        pars.append(params)
    return np.array(displays), np.array(pars)


if __name__ == "__main__":
    # training data
    train_dir = "./datasets/train/"
    test_dir = "./datasets/test/"

    for train_X in os.listdir(train_dir + "X/"):
        os.remove(train_dir + "X/" + train_X)
    for test_X in os.listdir(test_dir + "X/"):
        os.remove(test_dir + "X/" + test_X)

    for train_y in os.listdir(train_dir + "y/"):
        os.remove(train_dir + "y/" + train_y)
    for test_y in os.listdir(test_dir + "y/"):
        os.remove(test_dir + "y/" + test_y)

    for i in range(300):
        print(i)
        displays, params = create_dataset(256)
        np.savez_compressed(train_dir + "X/X{}.npz".format(i), displays)
        np.savez_compressed(train_dir + "y/y{}.npz".format(i), params)

    # testing data
    for i in range(300):
        print(i)
        displays, params = create_dataset(78)
        np.savez_compressed(test_dir + "X/X{}.npz".format(i), displays)
        np.savez_compressed(test_dir + "y/y{}.npz".format(i), params)
