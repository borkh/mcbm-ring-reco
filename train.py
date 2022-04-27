#!/usr/bin/env python3
import os, datetime, cv2, pickle
import tensorflow as tf
import numpy as np
from wandb.keras import WandbCallback
from create_data import *
from model import *
from sweep_configs import *
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow_addons.optimizers import Triangular2CyclicalLearningRate

import multiprocessing

def train_with_dataset(config=None):
    #multiprocessing.Queue(1000)
    # define model name ---------------------------------------------------------
    name, now = "200k", datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = "models/checkpoints/{}-{}.model".format(name, now)

    # define callbacks ----------------------------------------------------------
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3000)
    mc = tf.keras.callbacks.ModelCheckpoint(model_path,
                                            monitor="val_loss",
                                            save_best_only=True)
    # load data _______----------------------------------------------------------
    with open("data/200k.pkl", "rb") as f:
        x_train, y_train = pkl.load(f)
    # initialize agent ----------------------------------------------------------
    with wandb.init(config=None):
        wandb.run.log_code(".")
        config = wandb.config
        # create model ------------------------------------------------------------
        model = get_model(config.input_shape, config.output_shape, config)
        #model = tf.keras.models.load_model("models/checkpoints/200k-202204252345.model")
        #model = get_GAP_model(config.input_shape, config.output_shape, config)
        # compile model ---------------------------------------------------------
        vs = 0.3
        spe = x_train.shape[0]*(1-vs)/config.batch_size # calculate steps per epoch
        #lr = CosineDecayRestarts(config.max_lr, config.decay_length*spe, 1.0, config.lr_decay, config.init_lr)
        lr = Triangular2CyclicalLearningRate(config.init_lr, config.max_lr, config.decay_length*spe)
        #lr = 0.001
        opt = SGD(lr, momentum=0.9)
        model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])
        # fit model -------------------------------------------------------------
        model.fit(x_train, y_train, validation_split=vs,
                  epochs=config.epochs, batch_size=config.batch_size,
                  callbacks=[WandbCallback(), mc, es])

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project='ring-finder')
    wandb.agent(sweep_id, train_with_dataset, count=100)
