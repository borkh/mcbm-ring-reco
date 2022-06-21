#!/usr/bin/env python3
import datetime, os
import tensorflow as tf, pickle as pkl, numpy as np
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers import Triangular2CyclicalLearningRate

from create_data import *
from sweep_configs import *
from model import *
from one_cycle import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def train_with_dataset(conf=None):
    # define model name ---------------------------------------------------------
    name, now = "200k", datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = "models/checkpoints/{}-{}.model".format(name, now)

    # define callbacks ----------------------------------------------------------
    mc = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=False)
    # load data _______----------------------------------------------------------
    with open("data/200k.pkl", "rb") as f:
        x, y = pkl.load(f)
    # initialize agent ----------------------------------------------------------
    with wandb.init(config=None):
        wandb.run.log_code(".")
        conf = wandb.config
        # build model ------------------------------------------------------------
        model = build_model(conf.input_shape, conf.output_shape, conf)
        vs = 0.1
        spe = len(x) * (1-vs) / conf.batch_size # calculate steps per epoch
        steps = spe * conf.epochs

        # learning rate schedule ------------------------------------------------------------
        lr_schedule = OneCycleSchedule(conf.init_lr, conf.max_lr, steps, conf.mom_min, conf.mom_max, conf.phase0perc)

        # compile model ---------------------------------------------------------
        opt = SGD(conf.init_lr, momentum=0.95)
        model.compile(optimizer=opt, loss="mse", metrics=["acc"])

        # fit model -------------------------------------------------------------
        model.fit(x, y, validation_split=vs,
                  epochs=conf.epochs, batch_size=conf.batch_size,
                  callbacks=[WandbCallback(), mc, lr_schedule])

def train_with_generator(conf=None):
    try:
        # define model name ---------------------------------------------------------
        name, now = "generator", datetime.datetime.now().strftime("%Y%m%d%H%M")
        model_path = "models/checkpoints/{}-{}.model".format(name, now)

        # define callbacks ----------------------------------------------------------
        mc = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=False)
        # initialize agent ----------------------------------------------------------
        with wandb.init(config=None):
            wandb.run.log_code(".")
            conf = wandb.config
            # create generators -----------------------------------------------------------------
            traingen = DataGen(conf.input_shape, (conf.min_hits_per_ring, conf.max_hits_per_ring),
                                conf.ring_noise, conf.batch_size, conf.spe)
            # build model ------------------------------------------------------------
            model = build_model(conf.input_shape, conf.output_shape, conf)
            steps = conf.spe * conf.epochs

            # learning rate schedule ------------------------------------------------------------
            lr_schedule = OneCycleSchedule(conf.init_lr, conf.max_lr, steps, conf.mom_min, conf.mom_max, conf.phase0perc)

            # compile model ---------------------------------------------------------
            opt = SGD(conf.init_lr, momentum=0.95)
            model.compile(optimizer=opt, loss="mse", metrics=["acc"])

            # fit model -------------------------------------------------------------
            model.fit(traingen, steps_per_epoch=conf.spe, epochs=conf.epochs,
                      validation_data=traingen, validation_steps=int(conf.spe*0.1),
                      callbacks=[WandbCallback(), mc, lr_schedule])
            #lr_schedule.plot()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    sweep_id = wandb.sweep(run_config, project='ring-finder')
    wandb.agent(sweep_id, train_with_generator, count=1)
