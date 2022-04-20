#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts
from keras_lr_finder import LRFinder

def get_model(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    t = inputs
    #t = BatchNormalization()(inputs)

    for n in range(config.conv_layers):
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)), # double the number of filters in each subsequent layer
                   config.conv_kernel_size,
                   kernel_initializer="he_uniform",
                   padding=config.padding,
                   activation='relu',
                   name="block{}_conv0".format(n))(t)
        t = Dropout(config.conv_dropout)(t)

        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)),
                   config.conv_kernel_size,
                   kernel_initializer="he_uniform",
                   padding=config.padding,
                   activation='relu',
                   name="block{}_conv1".format(n))(t)
        t = Dropout(config.conv_dropout)(t)

        t = MaxPooling2D(config.pool_size, name="block{}_pool".format(n))(t)

    t = Flatten()(t)

    t = Dropout(config.fc_dropout)(t)
    t = Dense(config.fc_layer_size,
              kernel_initializer="he_uniform",
              activation="relu",
              name="fc")(t)

    t = Dropout(config.output_dropout)(t)
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

    for n in range(3):
        t = Conv2D(int(2 ** (np.log2(32) + n)), 3, padding="same",
                activation='relu')(t)
        t = Conv2D(int(2 ** (np.log2(32) + n)), 3, padding="same",
                activation='relu')(t)
        t = MaxPooling2D(2)(t)

    #t = Dropout(0.1)(t)
    t = Flatten()(t)
    for n in range(1):
        t = Dense(1024/(n+1), activation="relu", name="fc{}".format(n))(t)
    #t = Dropout(0.1)(t)
    outputs = Dense(output_shape, activation="relu", name="predictions")(t)

    model = Model(inputs, outputs)

    model.summary()
    return model

def get_model2(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    x = Conv2D(16, 3, padding="same", kernel_initializer="he_uniform", bias_initializer=Constant(0.001), activation="relu", name="block1_conv1")(inputs)
    x = Conv2D(32, 3, padding="same", kernel_initializer="he_uniform", bias_initializer=Constant(0.001), activation="relu", name="block1_conv2")(x)
    x = MaxPooling2D(2, name="block1_pool")(x)
    x = Conv2D(32, 3, padding="same", kernel_initializer="he_uniform", bias_initializer=Constant(0.001), activation="relu", name="block2_conv1")(x)
    x = Conv2D(64, 3, padding="same", kernel_initializer="he_uniform", bias_initializer=Constant(0.001), activation="relu", name="block2_conv2")(x)
    x = MaxPooling2D(2, name="block2_pool")(x)
    x = Conv2D(64, 3, padding="same", kernel_initializer="he_uniform", bias_initializer=Constant(0.001), activation="relu", name="block3_conv1")(x)
    x = Conv2D(128, 3, padding="same", kernel_initializer="he_uniform", bias_initializer=Constant(0.001), activation="relu", name="block3_conv2")(x)

    #x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu", name="dense")(x)

    #x = Dropout(0.1)(x)
    output = Dense(output_shape, activation="relu", name="predictions")(x)

    model = Model(inputs, output)

    model.summary()
    return model

def find_lr_range():
    ins, os, hpr, rn = (72,32,1), 15, (24, 33), 0.08
    gen = SynthGen(ins, os, hpr, rn)

    print("Training data...")
    x_train, y_train = gen.create_dataset(100000)

    x_test, y_test = gen.create_dataset(100)
    model = plain_net_wo_conf(ins, os)

    lr = 0.001
    opt= Adam(lr)
    model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])

    lr_finder = LRFinder(model)
    lr_finder.find(x_train, y_train, start_lr=1e-9, end_lr=1, batch_size=32, epochs=3)
    lr_finder.plot_loss(n_skip_end=1)
    plt.show()

if __name__ == "__main__":
    from create_data import *
    find_lr_range()
