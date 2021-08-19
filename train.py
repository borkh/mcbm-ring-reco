#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from model import *
from sweep_configs import *
from wandb.keras import WandbCallback
from itertools import permutations, chain

## ---------------------- load data ----------------------------
data_dir = "./datasets/"
displays = np.load(data_dir + "displays.npy", "r")
displays_test = np.load(data_dir + "displays_test.npy", "r")
params = np.load(data_dir + "params.npy", "r")
params_test = np.load(data_dir + "params_test.npy", "r")


def train_with_flip(config=None):
    with wandb.init(config=sweep_config):
        config = wandb.config
        model = create_model(displays[0].shape, params.shape[-1], config)

        model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy'])

        flipped_params = np.array(params)
        flipped = np.zeros((len(flipped_params), config.epochs))

        nof_epochs_wo_improvement = 0
        print("Epoch", 0)
        prev_history = model.fit(displays, flipped_params,
                                 batch_size=config.batch_size,
                                 validation_split=0.2,
                                 callbacks=[WandbCallback()])
        for epoch in range(1, config.epochs):
            print("Epoch", epoch)
            history = model.fit(displays, flipped_params,
                                batch_size=config.batch_size,
                                validation_split=0.2,
                                callbacks=[WandbCallback()])
            if history.history['val_accuracy'][0] - prev_history.history['val_accuracy'][0] < 0.001:
                nof_epochs_wo_improvement += 1
            prev_history = history

            pred_params = model.predict(displays)
            for i, (pred_rings, exp_rings) in enumerate(zip(pred_params, flipped_params)):
                # create all possible permutations for the parameters
                nested_exp_rings = [[exp_rings[i], exp_rings[i+1], exp_rings[i+2], exp_rings[i+3], exp_rings[i+4]] for i in range(0, len(exp_rings), 5)]
                perms = [list(chain(*i)) for i in list(permutations(nested_exp_rings))]

                mse = np.mean(np.square(pred_rings - exp_rings))
                mse_flipped = []
                for flipped_exp_rings in perms:
                    mse_flipped.append(np.mean(np.square(pred_rings - flipped_exp_rings)))

                min_mse, min_index = np.min(mse_flipped), np.argmin(mse_flipped)
                if min_mse < mse:
                    flipped_params[i] = perms[min_index]
                    flipped[i, epoch] = 1
            print("Flipped {} training samples ({} %)".format(
                            np.sum(flipped[:, epoch]),
                            np.mean(flipped[: epoch]) * 100.))
            # early stopping condition
            if nof_epochs_wo_improvement >= 5:
                break
#        model.save("models/32x2-CNN.model")


sweep_id = wandb.sweep(sweep_config, project='ellipses-params-finder')
#sweep_id = str("rhaas/params-finder-sweep/kr04o6qd")

wandb.agent(sweep_id, train_with_flip, count=1000)
