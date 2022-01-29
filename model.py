#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Add, Conv2D, Conv2DTranspose,
                                     BatchNormalization, Dropout, MaxPooling2D,
                                     AveragePooling2D, Flatten, Dense, Input,
                                     ReLU)
from tensorflow import Tensor


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int,
                   kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    num_filters = config.nof_initial_filters

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)

    outputs = Dense(output_shape, activation=config.fc_activation)(t)

    model = Model(inputs, outputs)
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy'])

    return  model

def create_plain_net(input_shape, output_shape, config=None):
    inputs = Input(input_shape)

    t = inputs
#    t = BatchNormalization()(inputs)

    for n in reversed(range(config.conv_layers)):
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)),
                   config.conv_kernel_size, padding=config.padding,
                   activation='relu')(t)
    #    t = BatchNormalization()(t)
        t = MaxPooling2D(config.pool_size, padding=config.padding)(t)

    t = Flatten()(t)
    t = Dense(config.fc_layer_size, activation="relu")(t)
    t = Dropout(config.dropout)(t)

    outputs = Dense(output_shape, activation=config.fc_activation)(t)

    model = Model(inputs, outputs)
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy'])

    return model

def create_simple_net(input_shape, output_shape, config=None):
    inputs = Input(input_shape)

    t = inputs
#    t = BatchNormalization()(inputs)

    for n in [32, 16, 10, 8]:
        t = Conv2D(n, config.conv_kernel_size, padding=config.padding,
                   activation='relu')(t)
    #    t = BatchNormalization()(t)
        t = MaxPooling2D(config.pool_size, padding=config.padding)(t)

    t = Flatten()(t)
    t = Dense(config.fc_layer_size, activation="relu")(t)
#    t = Dense(config.fc_layer_size/2, activation="relu")(t)
#    t = Dropout(config.dropout)(t)

    outputs = Dense(output_shape, activation=config.fc_activation)(t)

    model = Model(inputs, outputs)
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy'])

    return model
