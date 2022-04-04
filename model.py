#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Add, Conv2D, Conv2DTranspose,
                                     Dropout, MaxPooling2D, UpSampling2D,
                                     AveragePooling2D,
                                     Flatten, Dense, Input)
from tensorflow.keras.optimizers import Adam

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
    opt = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=opt, loss=config.loss, metrics=['accuracy'])

    return model

def deep_cnn(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    x = Conv2D(16, 3, padding="same", activation="relu", name="block1_conv1")(inputs)
    x = Conv2D(16, 3, padding="same", activation="relu", name="block1_conv2")(x)
    x = MaxPooling2D(2, padding="same", name="block1_pool")(x)
    x = Conv2D(32, 3, padding="same", activation="relu", name="block2_conv1")(x)
    x = Conv2D(32, 3, padding="same", activation="relu", name="block2_conv2")(x)
    x = MaxPooling2D(2, padding="same", name="block2_pool")(x)
    x = Conv2D(64, 3, padding="same", activation="relu", name="block3_conv1")(x)
    x = Conv2D(64, 3, padding="same", activation="relu", name="block3_conv2")(x)
    x = MaxPooling2D(2, padding="same", name="block3_pool")(x)

    x = Flatten()(x)
    x = Dense(1024, activation="relu", name="dense")(x)

    output = Dense(output_shape, activation="relu", name="predictions")(x)

    model = Model(inputs, output)
    model.compile(loss='mse', optimizer=Adam(learning_rate=config.learning_rate), metrics=["accuracy"])

    return model

def autoencoder(input_shape, output_shape, config=None):
    inputs = Input(input_shape)
    x = Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = UpSampling2D(2)(x)
    output = Conv2D(1, 3, padding="same", activation="relu")(x)

    model = Model(inputs, output, name="model")
    model.summary()
    opt = Adam(learning_rate=config.learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=["accuracy"])
    #model.fit(x_train, x_train, epochs=3, validation_split=0.1)

    return model

def encoder(input_shape, output_shape, cofig=None):
    inputs = Input(input_shape)
    x = Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(16, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(2, padding="same")(x)

    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    output = Dense(15, activation="relu")(x)

    model = Model(inputs, output)
    opt = Adam(learning_rate=config.learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=["accuracy"])

    return model
