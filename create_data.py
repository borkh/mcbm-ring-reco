#!/usr/bin/env python3
import os
import numpy as np
import random as rand
import pickle as pkl
from numpy.random import choice, laplace
from sklearn.datasets import make_circles, make_moons
from tqdm import tqdm
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
    def __init__(self, input_shape, output_shape, hits_per_ring, rn, batch_size=32, steps_per_epoch=1000):
        self.ins = input_shape
        self.os = output_shape
        self.bs = batch_size
        self.spe = steps_per_epoch
        self.minhits = hits_per_ring[0]
        self.maxhits = hits_per_ring[1]
        self.rn = rn

    def __getitem__(self, index):
        X = np.zeros((self.bs, self.ins[0], self.ins[1], self.ins[2]))
        Y = np.zeros((self.bs, self.os))
        for i in range(self.bs):
            x = Display(self.ins)
            x.add_ellipses(choice([0,1,2,3]), (self.minhits, self.maxhits), self.rn, choice(range(7,15)))
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
            x.add_ellipses(choice([0,1,2,3]), (self.minhits, self.maxhits), self.rn, choice(range(7,15)))
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
        obj.ee = 3
        obj.minX, obj.maxX, obj.minY, obj.maxY = (obj.ee,
                                                  obj.shape[0] - obj.ee,
                                                  obj.ee,
                                                  obj.shape[1] - obj.ee)
        obj.params = np.zeros(15)
        obj.positions = np.array([(x, y) for x in range(obj.shape[0]) for y in range(obj.shape[1])])
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

    def add_ellipses(self, nof_rings, hpr, rn=0, nof_noise_hits=None):
        indices = self.__get_indices(nof_rings)
        for n in range(nof_rings):
            yshift, xshift = indices[n] % self.shape[1], int(indices[n]/self.shape[1]) # shift the ellipses based on their index
            nod = 30 # number of hits that will be deleted
            hits, r = rand.randint(hpr[0] + nod, hpr[1] + nod), round(np.random.uniform(2.0,7.0), 1)
            X, y = make_circles(noise=rn, factor=.1, n_samples=(hits, 0))

            major, minor = r, r # create rings (major and minor used for possibilty of creating ellipses)
            X[:,0] *= major
            X[:,1] *= minor

            angle = 0 if major==minor else rand.randint(0, 90) # rotate ellipse
            X = rotate(X, angle)

            X = np.round(X, 0).astype('int32') # convert all entries to integers

            yshift, xshift = indices[n] % self.shape[1], int(indices[n]/self.shape[1]) # shift the ellipses based on their index
            X[:,0] += xshift
            X[:,1] += yshift

            # take only points that are in the display range in order to avoid having the majority of points outside
            # the array range and therefore fitting rings with only 2 or 3 points
            X = np.array([x for x in set(tuple(x) for x in X) & set(tuple(x) for x in self.positions)])

            X = X[:hits-nod] # delete a few pairs to have irregular distances between ring points

            for x, y in zip(X[:,0], X[:,1]): # set the values of the positions of the ring points in the display image to 1
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

def create_tree(file_, x, y, name):
    file_.cd()

    # Setup tree and branches with arrays
    tree = ROOT.TTree(name, name)
    tree.SetAutoSave(0)
    x_array = np.empty((72 * 32 * 1), dtype="float32")
    x_branch = tree.Branch("x", x_array, "x[{}]/F".format(72 * 32 * 1))

    y_array = np.empty((15), dtype="float32")
    y_branch = tree.Branch("y", y_array, "y[{}]/F".format(15))

    for x_, y_ in zip(x, y):
        # Reshape x_ to flat array
        x_ = x_.reshape(72 * 32 * 1)

        # Copy inputs and outputs to correct addresses
        x_array[:] = x_[:]
        y_array[:] = y_[:]   #np.argmax(y_)

        # Fill tree
        tree.Fill()

    tree.Write()

def create_root_file(x_train, y_train, path):
    # Convert dataset to ROOT file
    file_ = ROOT.TFile(path, "RECREATE")
    print(x_train.shape, y_train.shape)
    create_tree(file_, x_train, y_train, "train")
    file_.Write()
    file_.Close()

def create_datasets(size, path):
    """
    Create one dataset of size=size and save it two times in different
    formats (pickle and root formats)
    - pickle for tensorflow/keras training
    - root for ROOT/TMVA training

    path should be given as path to directory + filename without file extension
    (e.g. path="data/name")
    """
    ins, os, minhits, maxhits, rn = (72,32,1), 15, 12, 30, 0.01
    hpr = (minhits, maxhits)
    gen = SynthGen(ins, os, hpr, rn)
    x, y = gen.create_dataset(size)

    with open(path + ".pkl", "wb") as f:
        pkl.dump([x, y], f)

    root_path = path + ".root"
    create_root_file(x, y, root_path)

if __name__ == "__main__":
    size = 2000
    path = "data/" + str(int(size/1000)) + "k-noisy"
    create_datasets(size, path)

    # plot a few examples to check if the datasets were created and saved properly
    with open(path + ".pkl", "rb") as f:
        x, y = pkl.load(f)
    #events = np.array([plot_single_event(x[i], y[i]) for i in range(1000)])
    #for i in range(5):
    #    display_data(events, i)

    def show(M, N, indices=np.arange(1000)):
        fig, ax = plt.subplots(M,N)
        for n, m in zip(product(range(M), range(N)), indices):
            plot = plot_single_event(x[m], y[m])
            ax[n].imshow(plot)
        plt.show()
    show(3,4, indices=np.arange(1000)[:100])
    show(3,4, indices=np.arange(1000)[100:200])
    #show(4,8, indices=np.arange(1000)[200:300])
    #show(4,8, indices=np.arange(1000)[300:400])

    #events = np.array([plot_single_event(x[i], np.zeros(15)) for i in range(100)])
    #display_data(events)
    #plot_root(path + ".root")

    # TODO: - switch from random to np.random
    #       - remove hard coding of max ring scaling line 122-123
