import datetime
import os
import traceback

import absl.logging
import matplotlib.pyplot as plt

# suppress warnings and info messages
absl.logging.set_verbosity(absl.logging.ERROR)  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
os.environ['WANDB_SILENT'] = 'true'  # nopep8

import argparse

import cv2
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # nopep8
from keras_lr_finder import LRFinder
from tensorflow.keras.optimizers import SGD  # type: ignore
from wandb.keras import WandbCallback

import wandb
from data.create_data import DataGen
from models.model import *
from utils.one_cycle import *
from utils.utils import *


def lr_range_test(training_size=10000, start_lr=1e-7, end_lr=5, epochs=5) -> None:
    """
    Find the optimal learning rate range for the model.

    This function trains the model for a few epochs with a learning rate schedule
    that increases the learning rate exponentially from a very small value to a very large value.
    The learning rate is plotted against the loss, and the optimal learning rate range is determined
    based on the minimum value of the loss curve. The 'max_lr' and 'init_lr' values are then set
    in the run_config dictionary in sweep_configs.py. 'max_lr' should be set to the minimum value of
    the loss curve, and 'init_lr' should be set to about 1/20 to 1/100 of 'max_lr'.

    Parameters:
        start_lr (float, optional): The initial learning rate. Default is 1e-7.
        end_lr (float, optional): The final learning rate. Default is 5.
        epochs (int, optional): The number of epochs to train the model for. Default is 5.
    """
    os.environ['WANDB_MODE'] = 'dryrun'
    with wandb.init(config=None):  # type: ignore
        c = wandb.config

        train_gen = DataGen(train_dir, batch_size=training_size)
        x, y = train_gen[0]

        model = build_model(input_shape_, c)

        lr = 0.001
        opt = SGD(lr, momentum=0.95)

        model.compile(optimizer=opt, loss=custom_loss)

        lr_finder = LRFinder(model)
        lr_finder.find(x, y, start_lr=start_lr, end_lr=end_lr,
                       batch_size=c.batch_size, epochs=epochs)
        lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=3)
        plt.savefig('plots/lr_range_test.png')
        plt.show()


def train(c=None) -> None:
    """
    Trains a model using the specified hyperparameters and saves the model and
    training history to a specified directory.

    This function loads training and validation data generators, builds a model
    using a specified architecture, compiles the model with a mean squared error
    loss function and an SGD optimizer with a learning rate schedule based on
    the 1cycle policy, and trains the model. The model and training history are
    then saved to 'models/checkpoints/'.

    Parameters:
        c (dict, optional): A dictionary containing the hyperparameters for the
            model. The dictionary should include the following keys:
                - 'batch_size': The number of samples per gradient update.
                - 'epochs': The number of epochs to train the model for.
                - 'init_lr': The initial learning rate.
                - 'max_lr': The maximum learning rate.
                - 'conv_layers': The number of convolutional layers in the model.
                - 'nof_initial_filters': The number of filters in the first
                    convolutional layer.
                - 'conv_kernel_size': The size of the convolutional kernel.

    Returns:
        None
    """

    nof_files = len(os.listdir(train_dir + '/X'))
    name = f'{nof_files//1000000}M'
    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    model_path = f'models/checkpoints/{name}-{now}.model'

    with wandb.init(config=None):  # type: ignore
        c = wandb.config

        try:
            # define data generators for training and validation
            train_gen = DataGen(train_dir, batch_size=c.batch_size)
            val_gen = DataGen('data/val', batch_size=500)

            # calculate the number of steps per epoch and the total number of steps
            spe = train_gen.n // c.batch_size
            steps = spe * c.epochs

            lr_schedule = OneCycleSchedule(c.init_lr, c.max_lr, steps)

            model = build_model(input_shape_, c)

            opt = tf.keras.optimizers.SGD(
                c.init_lr, momentum=0.95, nesterov=True)
            # , run_eagerly=True)
            model.compile(optimizer=opt, loss=custom_loss)

            model.fit(train_gen,
                      validation_data=val_gen,
                      steps_per_epoch=spe,
                      epochs=c.epochs,
                      shuffle=True,
                      callbacks=[WandbCallback(), lr_schedule])

            model.save(model_path)

            lr_schedule.plot()

            X, y = val_gen[0]
            predictions, pred_time = predict(model, X)
            print(
                f'Inference took {pred_time}s to run. {pred_time / len(X)}s per event')
            fit_rings(X, predictions)

        except Exception:
            print(traceback.format_exc())


if __name__ == "__main__":
    try:
        __IPYTHON__  # type: ignore
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_dir', type=str, default='data/train',
                            help='The directory containing the training data.')
        parser.add_argument('--val_dir', type=str, default='data/val',
                            help='The directory containing the validation data.')
        parser.add_argument('--find_lr_range', action='store_true',
                            help='''Run the learning rate range finder to determine
                            the optimal learning rate range.''')
        args = parser.parse_args()

        train_dir = args.train_dir
        val_dir = args.val_dir
        find_lr_range = args.find_lr_range
    else:
        train_dir = 'data/train'
        val_dir = 'data/val'
        find_lr_range = False

    # load sample png form 'data/train' to get input shape
    input_shape_ = cv2.imread(
        train_dir + '/X/0.png')[..., :1].shape  # type: ignore

    if find_lr_range:
        sweep_id = wandb.sweep(run_config)
        wandb.agent(sweep_id, lr_range_test, count=1)
    else:
        sweep_id = wandb.sweep(run_config, project='ring-finder')
        wandb.agent(sweep_id, train, count=1)
