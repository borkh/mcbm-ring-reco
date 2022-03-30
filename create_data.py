#!/usr/bin/env python3
import numpy as np
from numpy.random import choice, laplace
import random as rand
from sklearn.datasets import make_circles
from tqdm import tqdm
import os
from visual_functions import *

"""
Adrains PhD Thesis:
    CBM:
        about 22 hits per ring
    mCBM:
        about 12-21 hits per ring
        radius: 2-7
"""

def rotate(img, angle=0):
    # transorm angle from deg to rad
    angle *= np.pi / 180
    rotation_matrix = np.array([np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]).reshape(2,2)
    return img @ rotation_matrix

class Display(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        obj = super().__new__(subtype, shape, dtype, buffer=np.zeros(shape),
                              offset=offset, strides=strides, order=order)
        obj.info = info
        obj.ee = 3 # value that determines how much the ring can extend over the edge of the display
        obj.minX, obj.maxX, obj.minY, obj.maxY = (obj.ee,
                                                  obj.shape[0] - obj.ee,
                                                  obj.ee,
                                                  obj.shape[1] - obj.ee)
        obj.params = np.zeros(15)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def __add_noise(self, nof_noise_hits=0):
        for _ in range(nof_noise_hits):
            x, y  = rand.randint(self.minX, self.maxX), rand.randint(self.minY, self.maxY)
            self[x,y] = 1

    def __get_indices(self, nof_rings):
        # uncomment to restrict ellipses to center of the display (no extension over the edges)
        #self[self.minX:self.maxX,self.minY:self.maxY] = 1 # set area where the centers of the ellipses are allowed to 1
        #indices = np.where(self.flatten() == 1)[0] # get the indices of that area
        # uncomment to have no restrictions of ellipses centers
        indices = range(self.flatten().shape[0])
        self[:,:] = 0

        return sorted(rand.choices(indices, k=nof_rings)) # return sorted list of size 'nof_rings' of random indices in that area


    def add_ellipses(self, nof_rings, nof_noise_hits=None):
        indices = self.__get_indices(nof_rings)
        for n in range(nof_rings):
            nof_hits = rand.randint(24, 42)
            X, y = make_circles(noise=.07, factor=.1, n_samples=(nof_hits, 0))
            X = X[:nof_hits-12] # delete a few pairs to make data look closer to real data
            #X = X[:int(nof_hits*0.7)] # delete a few pairs to make data look closer to real data

            #X = list(X)
            #X.sort(key=lambda x: x[0]+x[1])
            #X = np.array(X)

            #sb = rand.randint(0, int(nof_hits*0.4))
            #s = slice(sb, sb + int(nof_hits*0.6))
            #X = X[s] # delete a few pairs to make data look closer to real data
            r = round(np.random.uniform(3,7), 0)

            major, minor = r, r # create rings (major and minor used for possibilty of creating ellipses)
            X[:,0] *= major
            X[:,1] *= minor

            angle = 0 if major==minor else rand.randint(0, 90) # rotate ellipse
            X = rotate(X, angle)

            X = np.round(X, 0).astype('int32') # convert all entries to integers

            yshift, xshift = indices[n] % self.shape[1], int(indices[n]/self.shape[1]) # shift the ellipses based on their index
            X[:,0] += xshift
            X[:,1] += yshift

            for x, y in zip(X[:,0], X[:,1]): # set the values of the positions of the circles in the display image to 1
                if (x >= 0 and x < self.shape[0] and
                    y >= 0 and y < self.shape[1]):
                    self[x,y] = 1

            pars = [(xshift+0.5),
                    (yshift+0.5),
                    major,
                    minor,
                    angle] # write parameters of each rings into self.params

            for i in range(5):
                self.params[n*5 + i] = pars[i]

        if nof_noise_hits is not None:
            self.__add_noise(nof_noise_hits)

def create_dataset(batch_size):
    displays, pars = [], []
    for _ in range(batch_size):
        nof_rings = choice(np.array([1,2,3]))
        noise = choice(np.array([2,3]))

        display = Display((72,32,1))
        display.add_ellipses(nof_rings, noise)
        pars.append(display.params)

        #blank = np.zeros((72,40,1))
        #display = np.column_stack((display,blank)) # expand array to be in a squared shape
        #display = cv2.merge((display,display,display))
        displays.append(display)
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

    nof_batches = 1000
    batch_size = 512
    print("Creating training data...")
    for i in tqdm(range(nof_batches)):
        displays, params = create_dataset(batch_size)
        np.savez_compressed(train_dir + "X/X{}.npz".format(i), displays)
        np.savez_compressed(train_dir + "y/y{}.npz".format(i), params)

    print("Creating testing data...")
    for i in tqdm(range(nof_batches)):
        displays, params = create_dataset(int(batch_size*0.45))
        np.savez_compressed(test_dir + "X/X{}.npz".format(i), displays)
        np.savez_compressed(test_dir + "y/y{}.npz".format(i), params)
