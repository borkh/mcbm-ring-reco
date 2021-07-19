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

def create_model(input_shape, config=None):
    model = Sequential()
    model.add(InputLayer(input_shape))

    for n in range(config.conv_layers):
        model.add(Conv2D(config.conv_filters, config.conv_kernel_size, padding=config.padding, activation='relu'))
        model.add(BatchNormalization())
        if config.max_pooling:
            model.add(MaxPooling2D(config.pool_size, padding=config.padding))
        model.add(Dropout(config.dropout))

    model.add(Flatten())
    model.add(Dense(config.fc_layer_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(config.dropout))

    model.add(Dense(6, activation=config.fc_activation))

    return model
