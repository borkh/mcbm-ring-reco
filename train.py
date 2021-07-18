#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from model import *
from sweep_configs import *
from wandb.keras import WandbCallback

## ---------------------- load data ----------------------------
data_dir = "./datasets/"

# training data
displays = np.load(data_dir + "displays.npy", "r")
displays_test = np.load(data_dir + "displays_test.npy", "r")

# testing data
params = np.load(data_dir + "params.npy", "r")
params_test = np.load(data_dir + "params_test.npy", "r")


def train(config=None):
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=4)
    with wandb.init(config=sweep_config):
        config = wandb.config
        model = create_model(displays[0].shape, config)

        model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy'])
        model.fit(displays, params, batch_size=config.batch_size,
                  epochs=config.epochs, validation_split=0.2,
                  callbacks=[WandbCallback(), earlystop_callback])

#sweep_id = wandb.sweep(sweep_config, project='params-finder-sweep')
sweep_id = str("rhaas/params-finder-sweep/dneeyepq")

wandb.agent(sweep_id, train, count=2000)
