#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Add, Conv2D, Conv2DTranspose,
                                     BatchNormalization, Dropout, MaxPooling2D,
                                     AveragePooling2D, Flatten, Dense, Input,
                                     ReLU)
from tensorflow.keras.optimizers import Adam
from tensorflow import Tensor

def plain_net(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    t = inputs
    #t = BatchNormalization()(inputs)

    for n in range(config.conv_layers):
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)),
                   config.conv_kernel_size, padding=config.padding,
                   activation='relu', name="block{}_conv".format(n))(t)
        #t = BatchNormalization(name="block{}_batch_norm".format(n))(t)
        t = MaxPooling2D(config.pool_size, padding=config.padding, name="block{}_pool".format(n))(t)
        #t = Dropout(config.dropout, name="block{}_dropout".format(n))(t)

    t = Dropout(config.dropout)(t)
    t = Flatten()(t)
    for n in range(config.fc_layers):
        t = Dense(config.fc_layer_size/(n+1), activation="relu", name="fc{}".format(n))(t)
    t = Dropout(config.dropout)(t)
    outputs = Dense(output_shape, activation=config.fc_activation, name="predictions")(t)

    model = Model(inputs, outputs)
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy'])

    return model

def vgg16(input_shape, output_shape, config=None):
    vgg = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    x = Flatten()(vgg.output)
    x = Dense(output_shape, activation='sigmoid')(x)
    model = Model(vgg.input, x)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

    return model

def bottleneck_CNN(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    t = Conv2D(32, 3, padding="same", activation="relu")(inputs)
    t = Conv2D(64, 3, padding="same", activation="relu")(t)
    t = Conv2D(32, 3, padding="same", activation="relu")(t)
    t = MaxPooling2D((2,2), padding="same")(t)
    t = Conv2D(16, 3, padding="same", activation="relu")(t)
    t = Flatten()(t)
    t = Dropout(0.1)(t)
    t = Dense(1024, activation="relu")(t)
    output = Dense(output_shape, activation="sigmoid")(t)

    model = Model(inputs, output)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

    return model

def net(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    t = Conv2D(16, 9, padding="same", activation="relu")(inputs)
    t = MaxPooling2D((2,2), padding="same")(t)
    t = Conv2D(32, 7, padding="same", activation="relu")(t)
    t = MaxPooling2D((2,2), padding="same")(t)
    t = Conv2D(128, 5, padding="same", activation="relu")(t)
    t = MaxPooling2D((2,2), padding="same")(t)
    t = Conv2D(64, 5, padding="same", activation="relu")(t)

    t = Flatten()(t)
    t = Dropout(0.1)(t)
    t = Dense(1024, activation="relu")(t)
    output = Dense(output_shape, activation="relu")(t)

    model = Model(inputs, output)
    model.compile(loss='MeanSquaredError', optimizer="adam", metrics=["accuracy"])

    return model
