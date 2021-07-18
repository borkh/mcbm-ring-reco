#!/usr/bin/env python3
import numpy as np
import random as rand
from sklearn.datasets import make_circles
from tqdm import tqdm


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
        display, params = create_event(int(rand.uniform(2, 3)))
        displays.append(display)
        pars.append(params)
    return np.array(displays), np.array(pars)


if __name__ == "__main__":
    # training data
    displays, params = create_dataset(50000)

    data_dir = "./datasets/"
    np.save(data_dir + "displays.npy", displays)
    np.save(data_dir + "params.npy", params)

    # testing data
    displays, params = create_dataset(1000)

    data_dir = "./datasets/"
    np.save(data_dir + "displays_test.npy", displays)
    np.save(data_dir + "params_test.npy", params)
