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

def train_with_generator(config=None):
    # define model name ---------------------------------------------------------
    name, now = "PolyDecay", datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = "models/checkpoints/{}-{}.model".format(name, now)
    #load_path = "models/bmsf.model"
    # define callbacks ----------------------------------------------------------
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0., patience=12, baseline=1.5)
    mc = tf.keras.callbacks.ModelCheckpoint(model_path,
                                            monitor="loss",
                                            save_best_only=True)
    # initialize agent ----------------------------------------------------------
    with wandb.init(config=None):
        config = wandb.config
        traingen = SynthGen(config.input_shape,
                            config.output_shape,
                            (config.min_hits_per_ring, config.max_hits_per_ring),
                            config.ring_noise,
                            config.batch_size,
                            config.spe)
        # load model ------------------------------------------------------------
        try:
            print("Loading model {} and continuing training...\n".format(load_path))
            model = tf.keras.models.load_model(load_path)
            model.summary()
        except NameError:
            print("Creating model {} and starting training...\n".format(model_path))
            model = plain_net(config.input_shape, config.output_shape, config)
        # compile model ---------------------------------------------------------
        #lr = CosineDecayRestarts(config.max_lr, 12*config.spe, 2.0, config.decay, config.init_lr)
        lr = 0.001
        opt= Adam(lr)
        model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])
        # fit model -------------------------------------------------------------
        model.fit(traingen,
                  steps_per_epoch=config.spe,
                  epochs=config.epochs,
                  callbacks=[WandbCallback(), mc])

def train_with_dataset(config=None):
    # define model name ---------------------------------------------------------
    name, now = "200k", datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = "models/checkpoints/{}-{}.model".format(name, now)
    # define callbacks ----------------------------------------------------------
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30)
    mc = tf.keras.callbacks.ModelCheckpoint(model_path,
                                            monitor="loss",
                                            save_best_only=True)
    # initialize agent ----------------------------------------------------------
    with wandb.init(config=None):
        wandb.run.log_code(".")
        config = wandb.config
        with open("data/200k.pkl", "rb") as f:
            x_train, y_train = pkl.load(f)
        # create model ------------------------------------------------------------
        model = get_model(config.input_shape, config.output_shape, config)
        # compile model ---------------------------------------------------------
        vs = 0.2
        spe = x_train.shape[0]*(1-vs)/config.batch_size # calculate steps per epoch
        lr = CosineDecayRestarts(config.max_lr, config.decay_length*spe, 1.0, config.lr_decay, config.init_lr)
        opt= Adam(lr)
        model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])
        # fit model -------------------------------------------------------------
        model.fit(x_train, y_train, validation_split=vs,
                  epochs=config.epochs, batch_size=config.batch_size,
                  callbacks=[WandbCallback(), mc, es])

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project='ring-finder')
    wandb.agent(sweep_id, train_with_dataset, count=1000)
