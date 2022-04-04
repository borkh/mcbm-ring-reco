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

class SynthGen(tf.keras.utils.Sequence):
    def __init__(self, input_shape, output_shape, hpr, rn, batch_size=32, steps_per_epoch=1000):
        self.ins = input_shape
        self.os = output_shape
        self.bs = batch_size
        self.spe = steps_per_epoch
        self.hpr = hpr
        self.rn = rn

    def __getitem__(self, index):
        X = np.zeros((self.bs, self.ins[0], self.ins[1], self.ins[2]))
        Y = np.zeros((self.bs, self.os))
        for i in range(self.bs):
            x = Display(self.ins)
            x.add_ellipses(choice([0,1,2,3], p=[0.1,0.3,0.3,0.3]),
                           choice(self.hpr), self.rn, choice([5,6,7]))
            y = x.params
            X[i] += x
            Y[i] += y
        return X, Y

    def __len__(self):
        return self.spe

    def create_dataset(self, size=1000):
        X = np.zeros((size, self.ins[0], self.ins[1], self.ins[2]))
        Y = np.zeros((size, self.os))
        print("Creating dataset...")
        for i in tqdm(range(size)):
            x = Display(self.ins)
            x.add_ellipses(choice([0,1,2,3], p=[0.1,0.3,0.3,0.3]),
                           choice(self.hpr), self.rn, choice([5,6,7]))
            y = x.params
            X[i] += x
            Y[i] += y
        return X, Y

class Display(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        obj = super().__new__(subtype, shape, dtype, buffer=np.zeros(shape),
                              offset=offset, strides=strides, order=order)
        obj.info = info
        obj.ee = 2
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
        self[self.minX:self.maxX,self.minY:self.maxY] = 1 # set area where the centers of the ellipses are allowed to 1
        indices = np.where(self.flatten() == 1)[0] # get the indices of that area
        # uncomment to have no restrictions of ellipses centers
        #indices = range(self.flatten().shape[0])
        self[:,:] = 0

        return sorted(rand.choices(indices, k=nof_rings)) # return sorted list of size 'nof_rings' of random indices in that area

    def add_ellipses(self, nof_rings, hpr=30, rn=0, nof_noise_hits=None):
        indices = self.__get_indices(nof_rings)
        for n in range(nof_rings):
            X, y = make_circles(noise=rn, factor=.1, n_samples=(hpr, 0))
            X = X[:hpr-12] # delete a few pairs to make data look closer to real data
            r = round(np.random.uniform(2,8), 1)

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
    ins, os, hpr, rn = (72,32,1), 15, 30, 0.07
    gen = SynthGen(ins, os, hpr, rn)
    X, y = gen.create_dataset(batch_size)
    return X, y

if __name__ == "__main__":
    X, y = create_dataset(100)
    display_data(X)
    plt.show()
#    for i in range(3):
#        X = Display((72,32,1))
#        X.add_ellipses(choice([1,2,3]), 20, 0.08, choice([2,3]))
#        plt.imshow(X)
#        plt.show()
