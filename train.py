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
import ROOT
import cv2
from create_data import *

class SynthGen(tf.keras.utils.Sequence):
    def __init__(self, input_shape, output_shape, batch_size, steps_per_epoch):
        self.ins = input_shape
        self.os = output_shape
        self.bs = batch_size
        self.spe = steps_per_epoch

    def __getitem__(self, index):
        w, h, d = self.ins
        X = np.zeros((self.bs, w, h, d))
        Y = np.zeros((self.bs, self.os))
        for i in range(self.bs):
            x = Display(self.ins)
            x.add_ellipses(choice([0,1,2,3]), choice([3,4,5]))
            y = x.params
            X[i] += x
            Y[i] += y
        return X, Y

    def __len__(self):
        return self.spe

def train(config=None):
    now = datetime.datetime.now().strftime("%Y%m%d%H%M")
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0.001,
                                          patience=500)
    checkpoint_path = "models/checkpoints/{}".format(now)
    mc = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            monitor="val_loss",
                                            save_best_only=True)
    with wandb.init(config=sweep_config):
        config = wandb.config
        ## ------------- define model / continue training ------------------
        traingen = SynthGen(config.input_shape,
                            config.output_shape,
                            config.batch_size,
                            config.spe)
        testgen = SynthGen(config.input_shape,
                            config.output_shape,
                            config.batch_size,
                            int(config.spe*0.4))

        #model_path = "models/plain-mcbm-2022-03-29_2214.model"
        model_path = "models/{}.model".format(now)

        #print("Loading model {} and continuing training...\n".format(model_path))
        #model = tf.keras.models.load_model(model_path)

        model = plain_net(config.input_shape, config.output_shape, config)
        model.summary()
        model.fit(traingen,
                  steps_per_epoch=config.spe,
                  epochs=config.epochs,
                  validation_data=testgen,
                  validation_steps=int(config.spe*0.4),
                  callbacks=[WandbCallback(), es, mc])
#        print("Saving model {}...\n".format(model_path))
#        model.save(model_path)

if __name__ == "__main__":
    sweep_id = wandb.sweep(single_run_config, project='ring-finder')
    wandb.agent(sweep_id, train, count=1)

#    traingen = SynthGen((72,32,1), 15, 32, 32)
#    X, y = traingen.__getitem__(0)
#    for i in range(5):
#        plt.imshow(plot_single_event(X[i], y[i]))
#        plt.show()
