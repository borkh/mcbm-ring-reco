#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os, time
from model import *
from sweep_configs import *
from wandb.keras import WandbCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from itertools import permutations, chain
from tqdm import tqdm
import datetime

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.n = len(os.listdir(self.directory + "X"))
        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

        X = [np.load(self.directory + "X/X{}.npz".format(i), "r")['arr_0'] for i in indexes]
        y = [np.load(self.directory + "y/y{}.npz".format(i), "r")['arr_0'] for i in indexes]
        return np.array(X), np.array(y)

    def __len__(self):
        return int(self.n / self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(self.n)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def train(config=None):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=500)
    with wandb.init(config=sweep_config):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        config = wandb.config
        ## ---------------------- load data --------------------------------
        train_dir = "./datasets/train/"
        test_dir = "./datasets/test/"
        displays = np.load(train_dir + "X/X0.npz", "r")['arr_0']
        params = np.load(train_dir + "y/y0.npz", "r")['arr_0']
        nof_files = len(os.listdir(train_dir + "y"))

        ## ------------- define model / continue training ------------------
        #model_path = "models/plain-mcbm-2022-02-23_0914.model"
        model_path = "models/vgg16_mod-mcbm-2022-02-24_1437.model"
        #model_path = "models/vgg16_mod-mcbm-{}.model".format(now)

        #print("Loading model {} and continuing training...\n".format(model_path))
        model = tf.keras.models.load_model(model_path)

        #model = plain_net(displays.shape, params.shape[-1], config)
        #model = VGG16_mod(displays.shape, params.shape[-1], config)
        model.summary()

        traingen = CustomDataGen(train_dir, config.batch_size)
        testgen = CustomDataGen(test_dir, int(config.batch_size/3))
        spe = int(np.floor(nof_files/config.batch_size))

        model.fit(traingen,
                  steps_per_epoch=spe,
                  epochs=config.epochs,
                  validation_data=testgen,
                  callbacks=[WandbCallback(), es])
        print("Saving model {}...\n".format(model_path))
        model.save(model_path)


if __name__ == "__main__":
    sweep_id = wandb.sweep(single_run_config, project='ellipses-params-finder')
    wandb.agent(sweep_id, train, count=1)
