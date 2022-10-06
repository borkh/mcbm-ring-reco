#!/usr/bin/env python3
import datetime, os, cv2, wandb
import tensorflow as tf, pickle as pkl, numpy as np
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers import Adam, SGD

from utils.sweep_configs import *
from utils.one_cycle import *
from models.model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def train(conf=None):
    # define model name ---------------------------------------------------------
    name, now = "100k", datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = f"models/checkpoints/{name}-{now}.model"

    # define callbacks ----------------------------------------------------------
    mc = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=False)
    # load data _______----------------------------------------------------------
    x, y = np.load('data/100k.npz')['arr_0'], np.load('data/100k.npz')['arr_1']
    # initialize agent ----------------------------------------------------------
    with wandb.init(config=None):
        wandb.run.log_code(".")
        conf = wandb.config

        # build model ------------------------------------------------------------
        model = build_model2(x.shape[1:], y.shape, conf)
        #model = build_ae(x.shape[1:])
        vs = 0.1
        spe = len(x) / conf.batch_size # calculate steps per epoch
        steps = spe * conf.epochs

        # learning rate schedule ------------------------------------------------------------
        lr_schedule = OneCycleSchedule(conf.init_lr, conf.max_lr, steps, conf.mom_min, conf.mom_max, conf.phase0perc)

        # compile model ---------------------------------------------------------
        opt = SGD(conf.init_lr, momentum=0.95)
        #opt = Adam()
        model.compile(optimizer=opt, loss="mse", metrics=["acc"])

        # fit model -------------------------------------------------------------
        model.fit(x, y, validation_split=vs, epochs=conf.epochs, batch_size=conf.batch_size,
                  shuffle=True,
                  callbacks=[WandbCallback(), mc, lr_schedule])
        lr_schedule.plot()

if __name__ == "__main__":
    sweep_id = wandb.sweep(run_config, project='ring-finder')
    wandb.agent(sweep_id, train, count=1)
