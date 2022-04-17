#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Add, Conv2D, Conv2DTranspose,
                                     Dropout, MaxPooling2D, UpSampling2D,
                                     AveragePooling2D,
                                     Flatten, Dense, Input)
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts
from tensorflow.keras.initializers import Constant

def get_model(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    t = inputs
    #t = BatchNormalization()(inputs)

    for n in range(config.conv_layers):
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)),
                config.conv_kernel_size, kernel_initializer="he_uniform",
                bias_initializer=Constant(0.001), padding=config.padding,
                activation='relu', name="block{}_conv0".format(n))(t)
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)),
                config.conv_kernel_size, kernel_initializer="he_uniform",
                bias_initializer=Constant(0.001), padding=config.padding,
                activation='relu', name="block{}_conv1".format(n))(t)
        t = MaxPooling2D(config.pool_size, name="block{}_pool".format(n))(t)

    t = Dropout(config.dropout)(t)
    t = Flatten()(t)
    for n in range(config.fc_layers):
        t = Dense(config.fc_layer_size/(n+1), kernel_initializer="he_uniform",
                bias_initializer=Constant(0.001), activation="relu",
                name="fc{}".format(n))(t)
    t = Dropout(config.dropout)(t)
    outputs = Dense(output_shape, kernel_initializer="he_uniform",
            bias_initializer=Constant(0.001), activation=config.fc_activation,
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
