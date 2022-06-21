#!/usr/bin/env python3
import numpy as np, pickle as pkl
from numpy.random import choice
from sklearn.datasets import make_circles, make_moons
from tqdm import tqdm
from utils import *

import sys
np.set_printoptions(threshold=sys.maxsize)

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

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, input_shape, hits_per_ring, rn, batch_size=32, steps_per_epoch=1000):
        self.ins = input_shape
        self.bs = batch_size
        self.spe = steps_per_epoch
        self.minhits = hits_per_ring[0]
        self.maxhits = hits_per_ring[1]
        self.rn = rn

    def __getitem__(self, index):
        return self.create_dataset(self.bs)

    def __len__(self):
        return self.spe

    def create_dataset(self, size=1000):
        X = np.zeros((size, self.ins[0], self.ins[1], self.ins[2]))
        Y = list()
        Z = np.zeros((size))
        #B = np.zeros((size, 20, 5)) # ground truth boxes
        for i in range(size):# tqdm(range(size)):
            x = Display(self.ins)
            nof_rings = choice(range(0,4))
            x.add_ellipses(nof_rings, (self.minhits, self.maxhits), self.rn, choice(range(0,3)))
            y = x.params
            #b = x.bboxes
            X[i] += x
            Y.append(y)
            Z[i] += x.nof_rings
            #B[i] += b
        return X, np.array(Y)#).reshape((size,25))#, Z, B

class Display(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        obj = super().__new__(subtype, shape, dtype, buffer=np.zeros(shape),
                              offset=offset, strides=strides, order=order)
        obj.ee = 0
        obj.minX, obj.maxX, obj.minY, obj.maxY = (obj.ee,
                                                  obj.shape[0] - obj.ee,
                                                  obj.ee,
                                                  obj.shape[1] - obj.ee)
        obj.params = np.zeros((5,5))
        #obj.bboxes = np.zeros((4,5))
        obj.nof_rings = 0
        obj.positions = np.array([(x, y) for x in range(obj.shape[0])
                                         for y in range(obj.shape[1])])
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def __add_noise(self, nof_noise_hits=0):
        for _ in range(nof_noise_hits):
            x, y  = np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1])
            self[x,y] = 1

    def __get_indices(self, nof_rings):
        # uncomment to restrict ellipses to center of the display (no extension over the edges)
        self[self.minX:self.maxX,self.minY:self.maxY] = 1 # set area where the centers of the ellipses are allowed to 1
        indices = np.where(self.flatten() == 1)[0] # get the indices of that area
        # uncomment to have no restrictions of ellipses centers
        #indices = range(self.flatten().shape[0])
        self[:,:] = 0
        return sorted(choice(indices, size=nof_rings)) # return sorted list of size 'nof_rings' of random indices in that area

    def add_ellipses(self, nof_rings, hpr, rn=0, nof_noise_hits=None):
        indices = self.__get_indices(nof_rings)
        self.nof_rings = nof_rings
        n = 0
        while n < nof_rings:
            xshift, yshift = indices[n] % self.shape[1], int(indices[n]/self.shape[1]) # shift the ellipses based on their index
            nod = 20 # number of hits that will be deleted
            hits, r = np.random.randint(hpr[0] + nod, hpr[1] + nod), round(np.random.uniform(3.5,9.0), 1)
            X, y = make_circles(noise=rn, factor=.1, n_samples=(hits, 0))

            major, minor = r, r # create rings (major and minor used for possibilty of creating ellipses)
            X[:,1] *= major
            X[:,0] *= minor

            angle = 0 if major==minor else np.random.randint(0, 90) # rotate ellipse
            X = rotate(X, angle)
            X = np.round(X, 0).astype('int32') # convert all entries to integers

            X[:,1] += xshift
            X[:,0] += yshift

            # take only points that are in the display range so the majority of points aren't outside
            # the array range and therefore fitting rings with only 2 or 3 points
            X = X[:hits-nod] # delete a few pairs to have irregular distances between ring points
            X = np.array([x for x in set(tuple(x) for x in X) & set(tuple(x) for x in self.positions)]) # remove entries outside of display
            X = np.unique(X, axis=0) # only take unique pairs

            if len(X) >= hpr[0]: # check if rings still have at least 'minhits' hits
                for x, y in zip(X[:,0], X[:,1]): # set the values of the positions of the ring points in the display image to 1
                        self[x,y] = 1

                pars = [xshift+0.5, yshift+0.5, major, minor, angle]
                self.params[n] = np.array(pars) # write parameters of each ring into self.params

                #bbox = list(to_VOC_format(xshift+0.5, yshift+0.5, 2*major+1.5, 2*minor+1.5)) # write bbox of each ring into self.bboxes
                #if bbox[0] < 0: bbox[0] = 0
                #if bbox[1] < 0: bbox[1] = 0
                #if bbox[2] > self.shape[1]: bbox[2] = self.shape[1]
                #if bbox[3] > self.shape[0]: bbox[3] = self.shape[0]
                #self.bboxes[n] = np.array(bbox)
                n += 1
            else: # create new indices and create rings again
                indices = self.__get_indices(nof_rings)
                self.params = np.zeros((5,5))
                #self.bboxes = list()
                n = 0

        if nof_noise_hits is not None:
            self.__add_noise(nof_noise_hits)

def create_dataset(size, show_samples=True):
    path = f'data/{int(size/1000)}k'
    ins = (72,32,1)
    print("Creating datasets...")
    gen = DataGen(ins, (12, 25), 0.08)
    X, Y = gen.create_dataset(size)
    with open(path + ".pkl", "wb") as f:
        pkl.dump([X, Y], f)
    if show_samples:
        #Y = np.reshape(Y, (Y.shape[0], 5, Y.shape[1]//5))
        samples = [plot_single_event(x,y) for x, y in zip(X, Y)]
        display_images(1, 8, samples, 2)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    create_dataset(200)

    #with open(path + ".pkl", "rb") as f:
    #    x, y, z = pkl.load(f)
