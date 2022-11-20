#!/usr/bin/env python3
import sys
import os

import numpy as np
import plotly.express as px
import tensorflow as tf
from numpy.random import choice
from sklearn.datasets import make_circles
from tqdm import tqdm

sys.path.append('..')

from utils.utils import *  # nopep8


def rotate(img, angle=0):
    # transorm angle from deg to rad
    angle *= np.pi / 180
    rotation_matrix = np.array(
        [np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]).reshape(2, 2)
    return img @ rotation_matrix


class Display(np.ndarray):
    params: np.ndarray
    positions: np.ndarray

    def __new__(cls,
                shape,
                dtype=float,
                buffer=None,
                offset=0,
                strides=None,
                order=None,
                info=None):
        obj = super().__new__(Display,
                              shape,
                              dtype,
                              buffer=np.zeros(shape),
                              offset=offset,
                              strides=strides,
                              order=order)
        obj.params = np.zeros((5, 5))
        obj.nof_rings = 0
        obj.positions = np.array([(x, y) for x in range(obj.shape[0])
                                  for y in range(obj.shape[1])])
        obj.info = info
        return obj

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
        self[:, :] = 0  # set to zeroes in case of multiple calls
        indices = range(self.flatten().shape[0])

        return sorted(choice(indices, size=nof_rings))

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

            X = np.array(make_circles(
                noise=rn, factor=.1, n_samples=(hits, 0))[0])  # type: ignore

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


def add_to_dataset(dir_: str = 'test', n: int = 100, append: bool = True):
    """
    Add images and labels to dataset

    args:
        dir_: directory to save the dataset to
        n: number of images/labels to create
        append: if True, append to existing dataset
                if False, remove existing dataset and create new one

    hardcoded parameters:
        ins: input shape -> (width, height, channels) of event displays
        nof_rings: range of number of rings in event displays
        minhits, maxhits: range of number of hits in each ring
        ringnoise: noise in each ring -> how much the ring points are shifted from a perfect circle
    """

    ins = (72, 32, 1)
    minhits, maxhits = 12, 25
    ring_noise = 0.05

    range_ = range(0)
    if append:
        range_ = range(len(os.listdir(f'{dir_}/X')),
                       len(os.listdir(f'{dir_}/X')) + n)
    else:
        # remove files in dir_/X and dir_/y
        if input("Are you sure you want to delete the existing dataset? (y/n)") == 'y':
            for file in os.listdir(f'{dir_}/X'):
                os.remove(f'{dir_}/X/{file}')
            for file in os.listdir(f'{dir_}/y'):
                os.remove(f'{dir_}/y/{file}')
            range_ = range(n)

    for i in tqdm(range_):
        nof_rings = choice(range(1, 5))
        x = Display(ins)
        x.add_ellipses(nof_rings, (minhits, maxhits),
                       ring_noise, choice(range(1, 5)))
        y = np.array(x.params)

        cv2.imwrite(f'{dir_}/X/{i}.png', 255*x)
        np.savez_compressed(f'{dir_}/y/{i}.npz', y)
    print(f"Done. Created {n} images inside directory '{dir_}'.")


if __name__ == "__main__":
    add_to_dataset(dir_='test', n=1, append=True)

    # load dataset and inspect if it was created correctly
    X = np.array(
        [cv2.imread(f'val/X/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(10)])
    y = np.array([np.load(f'val/y/{i}.npz')['arr_0'] for i in range(10)])
    print(X.shape, y.shape)
    X = np.repeat(X[..., np.newaxis], 3, -1)

    imgs = np.array([plot_single_event(X[i], Y1=y[i]) for i in range(10)])
    px.imshow(imgs, animation_frame=0).show()
