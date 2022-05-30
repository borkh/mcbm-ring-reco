#!/usr/bin/env python3
import datetime, multiprocessing
import tensorflow as tf, pickle as pkl, numpy as np
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers import Triangular2CyclicalLearningRate

from create_data import *
from sweep_configs import *
from model import *
from one_cycle import *

def train_with_dataset(conf=None):
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
        conf = wandb.config
        # build model ------------------------------------------------------------
        model = build_model(conf.input_shape, conf.output_shape, conf)
        vs = 0.2
        steps = np.ceil(len(x_train) * (1-vs) / conf.batch_size) * conf.epochs
        spe = len(x_train) * (1-vs) / conf.batch_size # calculate steps per epoch

        # learning rate schedule ------------------------------------------------------------
        #lr = Triangular2CyclicalLearningRate(conf.init_lr, conf.max_lr, conf.decay_length * spe)
        lr_schedule = OneCycleSchedule(conf.init_lr, conf.max_lr, steps, conf.mom_min, conf.mom_max, conf.phase0perc)

        # compile model ---------------------------------------------------------
        #opt = SGD(lr, momentum=0.9)
        opt = SGD(conf.init_lr, momentum=0.95)
        model.compile(optimizer=opt, loss="mse", metrics=["acc"])

        # fit model -------------------------------------------------------------
        model.fit(x_train, y_train, validation_split=vs,
                  epochs=conf.epochs, batch_size=conf.batch_size,
                  callbacks=[WandbCallback(), mc, es, lr_schedule])
        #lr_schedule.plot()

def train_with_generator(conf=None):
    #multiprocessing.Queue(1000)
    # define model name ---------------------------------------------------------
    name, now = "generator", datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = "models/checkpoints/{}-{}.model".format(name, now)

    # define callbacks ----------------------------------------------------------
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3000)
    mc = tf.keras.callbacks.ModelCheckpoint(model_path,
                                            monitor="val_loss",
                                            save_best_only=True)
    # initialize agent ----------------------------------------------------------
    with wandb.init(config=None):
        wandb.run.log_code(".")
        conf = wandb.config
        # create generators -----------------------------------------------------------------
        traingen = SynthGen(conf.input_shape, conf.output_shape,
                            (conf.min_hits_per_ring, conf.max_hits_per_ring),
                            conf.ring_noise, conf.batch_size, conf.spe)
        # build model ------------------------------------------------------------
        model = build_model(conf.input_shape, conf.output_shape, conf)
        steps = conf.spe * conf.epochs

        # learning rate schedule ------------------------------------------------------------
        #lr = Triangular2CyclicalLearningRate(conf.init_lr, conf.max_lr, conf.decay_length * conf.spe)
        lr_schedule = OneCycleSchedule(conf.init_lr, conf.max_lr, steps, conf.mom_min, conf.mom_max, conf.phase0perc)

        # compile model ---------------------------------------------------------
        opt = SGD(conf.init_lr, momentum=0.95)
        model.compile(optimizer=opt, loss="mse", metrics=["acc"])

        # fit model -------------------------------------------------------------
        model.fit(traingen, steps_per_epoch=conf.spe, epochs=conf.epochs,
                  validation_data=traingen, validation_steps=int(conf.spe*0.1),
                  callbacks=[WandbCallback(), mc, es, lr_schedule])
        lr_schedule.plot()

if __name__ == "__main__":
    sweep_id = wandb.sweep(run_config, project='ring-finder')
    #sweep_id = str("rhaas/ring-finder/x6wx9dnm")
    wandb.agent(sweep_id, train_with_generator, count=1)
