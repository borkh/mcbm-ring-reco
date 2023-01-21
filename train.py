import datetime
import os
import sys
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
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from utils.utils import *  # nopep8
from utils.one_cycle import *  # nopep8
from models.model import *  # nopep8
from data.create_data import DataGen  # nopep8


def lr_range_test(start_lr=1e-7, end_lr=5, epochs=5) -> None:
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
    training_size = int(n_training_files * 0.05)
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

        plot_path = str(Path(root_dir, 'plots', 'lr_range_test.png'))
        plt.savefig(plot_path)
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

    name = f'{n_training_files//1000000}M'
    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    model_path = Path(root_dir, 'models', 'checkpoints',
                      f'{name}-{now}.model')

    with wandb.init(config=None):  # type: ignore
        c = wandb.config

        try:
            # define data generators for training and validation
            train_gen = DataGen(train_dir, batch_size=c.batch_size)
            val_gen = DataGen(val_dir, batch_size=32)

            # calculate the number of steps per epoch and the total number of steps
            spe = train_gen.n // c.batch_size
            steps = spe * c.epochs

            lr_schedule = OneCycleSchedule(c.init_lr, c.max_lr, steps)

            model = build_model(input_shape_, c)

            opt = tf.keras.optimizers.SGD(
                c.init_lr, momentum=0.95, nesterov=True)
            model.compile(optimizer=opt, loss=custom_loss)

            model.fit(train_gen,
                      validation_data=val_gen,
                      steps_per_epoch=spe,
                      epochs=c.epochs,
                      shuffle=True,
                      callbacks=[WandbCallback(), lr_schedule])

            model.save(model_path)

            lr_schedule.plot()

            X, _ = val_gen[0]
            predictions, _ = predict(model, X)
            fit_rings(X, predictions, title='Model Predictions on Validation Data', silent=silent)

        except Exception:
            print(traceback.format_exc())


if __name__ == "__main__":
    try:
        __IPYTHON__  # type: ignore
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_dir', type=str,
                            default=str(Path(root_dir, 'data', 'train')),
                            help='The directory containing the training data.')
        parser.add_argument('--val_dir', type=str,
                            default=str(Path(root_dir, 'data', 'val')),
                            help='The directory containing the validation data.')
        parser.add_argument('--find_lr_range', action='store_true',
                            help='''Run the learning rate range finder to determine
                            the optimal learning rate range.''')
        parser.add_argument('--silent', action='store_true',
                            help='''If set, plots will not be shown, but saved
                            to the plots directory.''')
        args = parser.parse_args()

        train_dir = Path(args.train_dir)
        val_dir = Path(args.val_dir)
        find_lr_range = args.find_lr_range
        silent = args.silent
    else:
        train_dir = Path(root_dir, 'data', 'train')
        val_dir = Path(root_dir, 'data', 'val')
        find_lr_range = True
        silent = False

    n_training_files = len(list(Path(train_dir, 'X').glob('*.png')))

    # load sample png form 'data/train/X' to get input shape
    sample_img_path = Path(train_dir, 'X', '0.png')
    print(sample_img_path)
    input_shape_ = cv2.imread(
        str(sample_img_path))[..., :1].shape  # type: ignore

    if find_lr_range:
        sweep_id = wandb.sweep(run_config)
        wandb.agent(sweep_id, lr_range_test, count=1)
    else:
        sweep_id = wandb.sweep(run_config, project='ring-finder')
        wandb.agent(sweep_id, train, count=1)
