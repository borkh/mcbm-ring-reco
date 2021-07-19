#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from model import *
from sweep_configs import *
from wandb.keras import WandbCallback

## ---------------------- load data ----------------------------
data_dir = "./datasets/"
displays = np.load(data_dir + "displays.npy", "r")
displays_test = np.load(data_dir + "displays_test.npy", "r")
params = np.load(data_dir + "params.npy", "r")
params_test = np.load(data_dir + "params_test.npy", "r")

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=4)

def train_with_flip(config=None):
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=4)
    with wandb.init(config=sweep_config):
        config = wandb.config
        model = create_model(displays[0].shape, config)

        model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy'])

        flipped_params = np.array(params)
        flipped = np.zeros((len(flipped_params), config.epochs))

        for epoch in range(config.epochs):
            print("Epoch", epoch)
            model.fit(displays, flipped_params, batch_size=config.batch_size,
                      validation_split=0.2, callbacks=[WandbCallback(),
                                                       earlystop_callback], verbose=2)
            pred_params = model.predict(displays)
            for i, (pred_rings, exp_rings) in enumerate(zip(pred_params, flipped_params)):
                flipped_exp_rings = np.concatenate([exp_rings[3:], exp_rings[:3]])

                mse = np.mean(np.square(pred_rings - exp_rings))
                mse_flipped = np.mean(np.square(pred_rings - flipped_exp_rings))

                if mse_flipped < mse:
                    flipped_params[i] = flipped_exp_rings
                    flipped[i, epoch] = 1
            print("Flipped {} training samples ({} %)".format(
                            np.sum(flipped[:, epoch]), 
                            np.mean(flipped[: epoch]) * 100.))
#        model.save("models/two_rings_32x3-CNN-params.model")

sweep_id = wandb.sweep(sweep_config, project='params-finder-sweep')
#sweep_id = str("rhaas/params-finder-sweep/rqiq6w7a")

wandb.agent(sweep_id, train_with_flip, count=1000)
