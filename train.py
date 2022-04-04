#!/usr/bin/env python3
import os, datetime, cv2
import tensorflow as tf
import numpy as np
from wandb.keras import WandbCallback
from itertools import permutations, chain
from tqdm import tqdm
from create_data import *
from model import *
from sweep_configs import *

def train(config=None):
    now = datetime.datetime.now().strftime("%Y%m%d%H%M")
    name = "rlrop"
    checkpoint_path = "models/checkpoints/{}-{}.model".format(name, now)
    mc = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            monitor="loss",
                                            save_best_only=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",
                                                factor=0.85,
                                                patience=1,
                                                min_delta=0.01)
    with wandb.init(config=None):
        config = wandb.config
        traingen = SynthGen(config.input_shape,
                            config.output_shape,
                            config.hits_per_ring,
                            config.ring_noise,
                            config.batch_size,
                            config.spe)

#        X, y = traingen.__getitem__(0)
#        display_data(X)
#        for i in range(5):
#            plt.imshow(plot_single_event(X[i], y[i]))
#            plt.show()

        #model_path = "models/rlrop-plain-202204031906.model"
        model_path = "models/{}-{}.model".format(name, now)

        #print("Loading model {} and continuing training...\n".format(model_path))
        #model = tf.keras.models.load_model(model_path)

        #model = plain_net(config.input_shape, config.output_shape, config)
        model = deep_cnn(config.input_shape, config.output_shape, config)
        model.summary()
        model.fit(traingen,
                  steps_per_epoch=config.spe,
                  epochs=config.epochs,
                  callbacks=[WandbCallback(), mc, rlrop])
        print("Saving model {}...\n".format(model_path))
        model.save(model_path)

if __name__ == "__main__":
    sweep_id = wandb.sweep(single_run_config, project='ring-finder')
    wandb.agent(sweep_id, train, count=1)
