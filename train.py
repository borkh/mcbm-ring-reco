import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import cv2
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback
from keras_lr_finder import LRFinder

import wandb
from models.model import *
from utils.one_cycle import *
from utils.sweep_configs import *


class dg(tf.keras.utils.Sequence):
    def __init__(self, dir_, batch_size=32):
        self.dir = tf.compat.as_str_any(dir_)
        self.bs = batch_size
        n = os.listdir(f'{self.dir}/X')
        self.n = len([file for file in n if file.endswith('.png')])

    def __getitem__(self, index):
        X = np.array([cv2.imread(f'{self.dir}/X/{i}.png', cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]/255.
                      for i in range(index*self.bs, (index+1)*self.bs)])
        y = np.array([np.load(f'{self.dir}/y/{i}.npy')
                      for i in range(index*self.bs, (index+1)*self.bs)])

        return X, y

    def __len__(self):
        return self.n//self.bs


def find_lr_range(x, y):
    model = build_model(x.shape[1:], config=None)

    lr = 0.001
    opt = SGD(lr)
    model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])

    lr_finder = LRFinder(model)
    lr_finder.find(x, y, start_lr=1e-7, end_lr=5, batch_size=100, epochs=5)
    lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
    plt.show()


def train(c=None):
    train_dir = 'data/train'
    nof_files = len(os.listdir(train_dir + '/X'))
    name, now = f'{nof_files//1000000}M', datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = f"models/checkpoints/{name}-{now}.model"
    mc = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss",
                                            save_best_only=False)

    with wandb.init(config=None):  # type: ignore
        c = wandb.config

        train_dg = dg(train_dir, batch_size=c.batch_size)
        val_dg = dg('data/val', batch_size=c.batch_size)

        spe = nof_files // c.batch_size  # steps per epoch
        steps = spe * c.epochs

        lr_schedule = OneCycleSchedule(c.init_lr, c.max_lr, steps,
                                       c.mom_min, c.mom_max, c.phase0perc)

        model = build_model((72, 32, 1), c)

        opt = tf.keras.optimizers.SGD(c.init_lr, momentum=0.95)
        model.compile(optimizer=opt, loss="mse", metrics=["acc"])

        model.fit(train_dg,
                  validation_data=val_dg,
                  steps_per_epoch=spe,
                  epochs=c.epochs,
                  shuffle=True,
                  callbacks=[WandbCallback(), mc, lr_schedule])


if __name__ == "__main__":
    sweep_id = wandb.sweep(run_config, project='ring-finder')
    wandb.agent(sweep_id, train, count=1)
