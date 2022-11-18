#!/usr/bin/env python3
import pickle as pkl
import sys
import time

import numpy as np
import plotly.express as px
import tensorflow as tf
from numpy.random import choice
from sklearn.datasets import make_circles
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)
sys.path.append('../')
from utils.utils import *  # nopep8


def rotate(img, angle=0):
    # transorm angle from deg to rad
    angle *= np.pi / 180
    rotation_matrix = np.array(
        [np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]).reshape(2, 2)
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
        X: np.ndarray = np.zeros((size, self.ins[0], self.ins[1], self.ins[2]))
        Y = list()
        Z = np.zeros((size))
        for i in tqdm(range(size)):
            x = Display(self.ins)
            nof_rings = choice(range(2, 4))
            x.add_ellipses(nof_rings, (self.minhits, self.maxhits),
                           self.rn, choice(range(1, 3)))
            y = x.params
            X[i] += x
            Y.append(y)
            Z[i] += x.nof_rings
        return X, np.array(Y)


class Display(np.ndarray):
    ee: float
    minX: float
    maxX: float
    minY: float
    maxY: float
    params: np.ndarray
    positions: np.ndarray

    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        self = super().__new__(Display, shape, dtype, buffer=np.zeros(shape),
                               offset=offset, strides=strides, order=order)
        self.ee = 0
        self.minX, self.maxX, self.minY, self.maxY = (self.ee,
                                                      self.shape[0] - self.ee,
                                                      self.ee,
                                                      self.shape[1] - self.ee)
        self.params = np.zeros((5, 5))
        self.nof_rings = 0
        self.positions = np.array([(x, y) for x in range(self.shape[0])
                                  for y in range(self.shape[1])])
        self.info = info
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

    def __add_noise(self, nof_noise_hits=0):
        for _ in range(nof_noise_hits):
            x, y = np.random.randint(
                0, self.shape[0]), np.random.randint(0, self.shape[1])
            self[x, y] = 1

    def __get_indices(self, nof_rings):
        # uncomment to restrict ellipses to center of the display (no extension over the edges)
        # set area where the centers of the ellipses are allowed to 1
        self[self.minX:self.maxX, self.minY:self.maxY] = 1
        # get the indices of that area
        indices = np.where(self.flatten() == 1)[0]
        # uncomment to have no restrictions of ellipses centers
        #indices = range(self.flatten().shape[0])
        self[:, :] = 0

        if nof_rings > 1:
            while True:
                # return sorted list of size 'nof_rings' of random indices in that area
                ids = sorted(choice(indices, size=nof_rings))
                break_crit = 0

                for n in range(nof_rings-1):
                    # shift the ellipses based on their index
                    x2, x1 = divmod(ids[n], self.shape[1])
                    # shift the ellipses based on their index
                    y2, y1 = divmod(ids[n+1], self.shape[1])
                    x = np.array([x1, x2])
                    y = np.array([y1, y2])
                    # calculate the euclidean distance between the centers of the ellipses
                    d = np.linalg.norm(x-y)

                    if d >= 2:
                        break_crit += 1

                if break_crit < nof_rings - 1:
                    time.sleep(1)
                else:
                    break
        else:
            # return sorted list of size 'nof_rings' of random indices in that area
            ids = sorted(choice(indices, size=nof_rings))

        return ids

    def add_ellipses(self, nof_rings: int, hpr: tuple, rn: float = 0, nof_noise_hits: int = 0):
        indices = self.__get_indices(nof_rings)
        self.nof_rings = nof_rings
        n = 0
        while n < nof_rings:
            # shift the ellipses based on their index
            yshift, xshift = divmod(indices[n], self.shape[1])
            nod = 20  # number of hits that will be deleted

            hits: int = np.random.randint(hpr[0] + nod, hpr[1] + nod)
            r: int = np.round(np.random.uniform(4.0, 8.0), 1)

            X: np.ndarray = np.array(make_circles(
                noise=rn, factor=.1, n_samples=(hits, 0))[0])

            # create rings (major and minor used for possibilty of creating ellipses)
            major, minor = r, r
            X[:, 1] *= major
            X[:, 0] *= minor

            angle = 0 if major == minor else np.random.randint(
                0, 90)  # rotate ellipse
            X = rotate(X, angle)
            # convert all entries to integers
            X = np.round(X, 0).astype('int32')

            X[:, 1] += xshift
            X[:, 0] += yshift

            # take only points that are in the display range so the majority of points aren't outside
            # the array range and therefore fitting rings with only 2 or 3 points
            # delete a few pairs to have irregular distances between ring points
            X = X[:hits-nod]
            X = np.array([x for x in set(tuple(x) for x in X) & set(
                tuple(x) for x in self.positions)])  # remove entries outside of the display
            X = np.unique(X, axis=0)  # only take unique pairs

            if len(X) >= hpr[0]:  # check if rings still have at least 'minhits' hits
                # set the values of the positions of the ring points in the display image to 1
                for x, y in zip(X[:, 0], X[:, 1]):
                    self[x, y] = 1

                pars = [xshift+0.5, yshift+0.5, major, minor, angle]
                # write parameters of each ring into self.params
                self.params[n] = np.array(pars)

                n += 1
            else:  # create new indices and create rings again
                indices = self.__get_indices(nof_rings)
                self.params = np.zeros((5, 5))
                n = 0

        if nof_noise_hits != 0:
            self.__add_noise(nof_noise_hits)


def create_dataset(size, name="", save=True, show_samples=True):
    ins = (72, 32, 1)
    print("Creating dataset...")
    gen = DataGen(ins, (12, 25), 0.05)
    X, y = gen.create_dataset(size)

    if show_samples:
        # transform X to rgb in order to show coloured ring fits
        X = np.repeat(X, 3, axis=3)
        samples = np.array([plot_single_event(x, y) for x, y in zip(X, y)])
#        fig = px.imshow(samples,
#                        binary_string=True,
#                        facet_col=0,
#                        facet_col_wrap=5,
#                        height=1500)
        fig = px.imshow(samples, animation_frame=0, height=800)
        fig.update_yaxes(tickmode="array",
                         tickvals=np.array([200, 400, 600]),
                         ticktext=np.array([20, 40, 60]))
        fig.update_xaxes(tickmode="array",
                         tickvals=np.array([100, 300]),
                         ticktext=np.array([10, 30])).show()

    if save:
        with open(f'{int(size/1000)}k-{name}.pkl', 'wb') as f:
            pkl.dump([X, y], f)


if __name__ == "__main__":
    create_dataset(20, "fixed19")
