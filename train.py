#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os, time
from model import *
from sweep_configs import *
from wandb.keras import WandbCallback
from itertools import permutations, chain
from tqdm import tqdm
import datetime

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, directory):
        self.n = nof_files
        self.directory = directory

    def on_epoch_end(self):
        pass
        """
        print("\nFlipping parameters in ", self.directory)
        flipped_total = []
        global model
        for i in tqdm(range(self.n)):
            nameX = self.directory + "X/X{}.npz".format(i)
            namey = self.directory + "y/y{}.npz".format(i)

            X = np.load(nameX, "r")['arr_0']
            y = np.load(namey, "r")['arr_0']
            pars, flipped = flip_pars(model, X, y)
            flipped_total.append(flipped)
            np.savez_compressed(namey, pars)
        print("Flipped {} training samples of {} total samples ({} %)".format(np.sum(flipped_total),
                                                                              self.n * y.shape[0],
                                                                              100 * np.sum(flipped_total) / (self.n * y.shape[0])))
        """

    def __getitem__(self, index):
        X = np.load(self.directory + "X/X{}.npz".format(index), "r")['arr_0']
        y = np.load(self.directory + "y/y{}.npz".format(index), "r")['arr_0']
        return X, y

    def __len__(self):
        return self.n

def flip_pars(model, disps, pars):
    pred_pars = model.predict(disps)
    flipped = np.zeros((len(pars)))

    for i, (pred_rings, exp_rings) in enumerate(zip(pred_pars, pars)):
        # create all possible permutations for the parameters
        nested_exp_rings = [[exp_rings[i], exp_rings[i+1],
                             exp_rings[i+2], exp_rings[i+3],
                             exp_rings[i+4]] for i in range(0, len(exp_rings),
                                                            5)]
        perms = [list(chain(*i)) for i in list(permutations(nested_exp_rings))]

        mse = np.mean(np.square(pred_rings - exp_rings))
        mse_flipped = []
        for flipped_exp_rings in perms:
            mse_flipped.append(np.mean(np.square(pred_rings - flipped_exp_rings)))

        min_mse, min_index = np.min(mse_flipped), np.argmin(mse_flipped)
        if min_mse < mse:
            pars[i] = perms[min_index]
            flipped[i] = 1

    return pars, np.sum(flipped[:])

def train_with_flip(config=None):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=500)
    with wandb.init(config=sweep_config):
        global displays
        global params
        global model
        config = wandb.config

        #model_path = "models/large_kernel-mcbm-2022-02-02_1406.model"
        model_path = "models/plain-mcbm-{}.model".format(now)

        #print("Loading model {} and continuing training...\n".format(model_path))
        #model = tf.keras.models.load_model(model_path)

        model = plain_net(displays[0].shape, params.shape[-1], config)
        #model = simple_net(displays[0].shape, params.shape[-1], config)
        #model = VGG16_mod(displays[0].shape, params.shape[-1], config)
        model.summary()

        model.fit(datagen, steps_per_epoch=nof_files, epochs=config.epochs,
                  validation_data=testgen, callbacks=[WandbCallback(),
                                                      es])
        print("Saving model {}...\n".format(model_path))
        model.save(model_path)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    ## ---------------------- load data --------------------------------
    train_dir = "./datasets/train/"
    test_dir = "./datasets/test/"
    nof_files = len(os.listdir(train_dir + "y/"))

    displays = np.load(train_dir + "X/X0.npz", "r")['arr_0']
    params = np.load(train_dir + "y/y0.npz", "r")['arr_0']

    datagen = CustomDataGen(train_dir)
    testgen = CustomDataGen(test_dir)

#    sweep_id = wandb.sweep(sweep_config, project='ellipses-params-finder')
    sweep_id = wandb.sweep(single_run_config, project='ellipses-params-finder')
#    sweep_id = str("rhaas/ellipses-params-finder/4xlmkw8y")

    wandb.agent(sweep_id, train_with_flip, count=1)
