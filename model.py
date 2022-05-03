#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts
from keras_lr_finder import LRFinder
from tensorflow.keras.regularizers import l1, l2

def get_model(input_shape, output_shape, config=None):
    reg = l2(0.001)

    inputs = Input(input_shape)
    t = inputs

    for n in range(config.conv_layers):
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)), # double the number of filters in each subsequent layer
                   config.conv_kernel_size,
                   kernel_initializer="he_uniform",
                   padding="same",
                   activation="relu")(t)
        t = BatchNormalization()(t)
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)), # double the number of filters in each subsequent layer
                   config.conv_kernel_size,
                   kernel_initializer="he_uniform",
                   padding="same",
                   activation="relu")(t)
        t = BatchNormalization()(t)
        t = MaxPooling2D((2,2), padding="same")(t)

    t = Flatten()(t)
    t = BatchNormalization()(t)
    t = Dense(config.fc_layer_size, kernel_initializer="he_uniform", activation="relu")(t)
    t = BatchNormalization()(t)
    outputs = Dense(output_shape,
                    kernel_initializer="he_uniform",
                    activation="relu", name="predictions")(t)

    model = Model(inputs, outputs)
    model.summary()
    return model

def get_GAP_model(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    t = inputs

    for n in range(config.conv_layers):
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)),
                   config.conv_kernel_size,
                   kernel_initializer="he_uniform",
                   padding="same",
                   activation='relu')(t)
        t = BatchNormalization()(t)
        t = MaxPooling2D(config.pool_size, padding="same")(t)

    t = GlobalAveragePooling2D()(t)
    t = BatchNormalization()(t)

    outputs = Dense(output_shape,
                    kernel_initializer="he_uniform",
                    activation=config.fc_activation,
                    name="predictions")(t)

    model = Model(inputs, outputs)
    model.summary()
    return model

def get_model_no_config(input_shape, output_shape):
    inputs = Input(input_shape)
    t = inputs

    for n in range(4):
        t = Conv2D(int(2 ** (np.log2(64) + n)), 3, padding="same",
                activation='relu')(t)
        t = BatchNormalization()(t)
        t = Conv2D(int(2 ** (np.log2(64) + n)), 3, padding="same",
                activation='relu')(t)
        t = BatchNormalization()(t)
        t = MaxPooling2D(2, padding="same")(t)

    t = Flatten()(t)
    t = BatchNormalization()(t)
    t = Dense(64, activation="relu", name="fc{}".format(n))(t)
    t = BatchNormalization()(t)
    outputs = Dense(output_shape, activation="relu", name="predictions")(t)

    model = Model(inputs, outputs)

    model.summary()
    return model

def get_model2(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    x = Conv2D(8, 7, padding="same", kernel_initializer="he_uniform", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, padding="same")(x)

    x = Conv2D(16, 5, padding="same", kernel_initializer="he_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, padding="same")(x)

    x = Conv2D(32, 3, padding="same", kernel_initializer="he_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, padding="same")(x)

    x = Conv2D(64, 3, padding="same", kernel_initializer="he_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, padding="same")(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)

    output = Dense(output_shape, activation="relu", name="predictions")(x)

    model = Model(inputs, output)

    model.summary()
    return model

def find_lr_range():
    ins, os, hpr, rn = (72,32,1), 15, (24, 33), 0.08
    with open("data/200k.pkl", "rb") as f:
        x_train, y_train = pkl.load(f)
    model = get_model_no_config(ins, os)

    lr = 0.05
    opt= SGD(lr)
    model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])

    lr_finder = LRFinder(model)
    lr_finder.find(x_train, y_train, start_lr=1e-7, end_lr=1, batch_size=256, epochs=3)
    lr_finder.plot_loss(n_skip_end=1)
    plt.show()

if __name__ == "__main__":
    from create_data import *
    find_lr_range()
