#!/usr/bin/env python3
import datetime
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v1 import SGD
from wandb.keras import WandbCallback

import wandb
from models.model import *
from utils.one_cycle import *
from utils.sweep_configs import *


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, dir_, batch_size=32):
        self.dir = dir_
        self.bs = batch_size
        self.n = len(os.listdir(f'{self.dir}/X'))

    def __getitem__(self, index):
        X = np.array([cv2.imread(f'{self.dir}/X/{i}.png')
                      for i in range(index*self.bs, (index+1)*self.bs)])
        y = np.array([np.load(f'{self.dir}/y/{i}.npz')['arr_0']
                      for i in range(index*self.bs, (index+1)*self.bs)])

        return X, y

    def __len__(self):
        return self.n//self.bs


def train(c=None):
    name, now = "900k", datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = f"models/checkpoints/{name}-{now}.model"
    mc = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss",
                                            save_best_only=False)

    with wandb.init(config=None):  # initialize wandb agent
        c = wandb.config

        train_gen = DataGen('data/train/', c.batch_size)
        #val_gen = DataGen('data/val', c.batch_size)

        nof_files = len(os.listdir('data/train/X'))
        spe = nof_files / c.batch_size  # steps per epoch
        steps = spe * c.epochs

        lr_schedule = OneCycleSchedule(c.init_lr,
                                       c.max_lr,
                                       steps,
                                       c.mom_min,
                                       c.mom_max,
                                       c.phase0perc)

        model = build_model((72, 32, 3), (5, 5), c)

        opt = SGD(c.init_lr, momentum=0.95)
        model.compile(optimizer=opt, loss="mse", metrics=["acc"])

        model.fit(train_gen,
                  steps_per_epoch=spe,
                  epochs=c.epochs,
                  shuffle=True,
                  callbacks=[WandbCallback(), mc, lr_schedule])


if __name__ == "__main__":
    sweep_id = wandb.sweep(run_config, project='ring-finder')
    wandb.agent(sweep_id, train, count=1)
