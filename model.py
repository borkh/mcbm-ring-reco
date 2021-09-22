#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
                                     BatchNormalization, Dropout, MaxPooling2D,
                                     Flatten, Dense)

def custom_loss_function(y_true, y_pred):
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)

def create_model(input_shape, output_shape, config=None):
    model = Sequential()
    model.add(InputLayer(input_shape))

    for n in range(config.conv_layers):
        model.add(Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)), config.conv_kernel_size,
                         padding=config.padding, activation='relu'))
        model.add(BatchNormalization())
        if config.max_pooling:
            model.add(MaxPooling2D(config.pool_size, padding=config.padding))
        model.add(Dropout(config.dropout))

    model.add(Flatten())
    model.add(Dense(config.fc_layer_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(config.dropout))

    model.add(Dense(output_shape, activation=config.fc_activation))

    return model

def create_model2(input_shape, output_shape):
    model = Sequential()
    model.add(InputLayer(input_shape))

    model.add(Conv2D(32, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(output_shape, activation='relu'))

    return model
