#!/usr/bin/env python3
import os, datetime, cv2
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from wandb.keras import WandbCallback
from create_data import *
from model import *
from sweep_configs import *
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
from tensorflow.keras.optimizers.schedules import *
from tensorflow_addons.optimizers import *

def train(config=None):
    #----------------------------------------------------------------------------
    # define model name ---------------------------------------------------------
    name, now = "CyclicalLearningRate", datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = "models/checkpoints/{}-{}.model".format(name, now)
    #load_path = "models/bmsf.model"
    #----------------------------------------------------------------------------
    # define callbacks ----------------------------------------------------------
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=1000)
    mc = tf.keras.callbacks.ModelCheckpoint(model_path,
                                            monitor="loss",
                                            save_best_only=True)
    #----------------------------------------------------------------------------
    # training ------------------------------------------------------------------
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
            #model = deep_cnn(config.input_shape, config.output_shape, config)
        #------------------------------------------------------------------------
        # compile model ---------------------------------------------------------
        lr = CosineDecayRestarts(config.max_lr, config.epochs*config.spe, 1.0, config.decay, config.init_lr)
        #lr = CosineDecayRestarts(config.max_lr, 4*config.spe, 1.0, config.decay, config.init_lr)
        #lr = Triangular2CyclicalLearningRate(config.init_lr, config.max_lr, step_size=2*config.spe)
        #lr = config.learning_rate
        opt= Adam(lr)
        model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])
        #------------------------------------------------------------------------
        # fit model -------------------------------------------------------------
        model.fit(traingen,
                  steps_per_epoch=config.spe,
                  epochs=config.epochs,
                  callbacks=[WandbCallback(), mc, es])
        #------------------------------------------------------------------------
        # recompile model -------------------------------------------------------
        #lr = CosineDecayRestarts(0.0001, 2000, 2.0, 0.9, 0.0)
        #model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])
        #------------------------------------------------------------------------
        # fit model again -------------------------------------------------------
        #model.fit(traingen,
        #          steps_per_epoch=config.spe,
        #          epochs=config.epochs,
        #          callbacks=[WandbCallback(), mc, plr])

if __name__ == "__main__":
    sweep_id = wandb.sweep(single_run_config, project='ring-finder')
    wandb.agent(sweep_id, train, count=1)

    from keras_lr_finder import LRFinder
    def debug():
        ins, os, hpr, rn = (72,32,1), 15, (24, 33), 0.08
        gen = SynthGen(ins, os, hpr, rn)

        print("Training data...")
        x_train, y_train = gen.create_dataset(160000)

        x_test, y_test = gen.create_dataset(100)
        model = plain_net_wo_conf(ins, os)

        #lr = CyclicalLearningRate(config.init_lr, config.max_lr, scale_fn=lambda x: 1/(2.**(x-1)), step_size=3*config.spe)
        lr = 0.001
        opt= Adam(lr)
        model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])

        lr_finder = LRFinder(model)
        lr_finder.find(x_train, y_train, start_lr=1e-9, end_lr=1, batch_size=32, epochs=3)
        lr_finder.plot_loss(n_skip_end=1)
        plt.show()

#        model.fit(x_train, y_train, epochs=10, batch_size=32,
#                validation_split=0.2, callbacks=[])#, rlrop])
#    debug()
