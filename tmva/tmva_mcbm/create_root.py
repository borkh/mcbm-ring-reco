#!/usr/bin/env python3
import ROOT
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from create_data import *

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

if __name__ == "__main__":
    print("Creating training data...")
    X_train, y_train = create_dataset(700000)
    print("Creating testing data...")
    X_test, y_test = create_dataset(300000)

    # Convert dataset to ROOT file
    file_ = ROOT.TFile("test.root", "RECREATE")
    create_tree(file_, X_train, y_train, "train")
    create_tree(file_, X_test, y_test, "test")
    #file_.Write()
    file_.Close()