#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer, Conv2D, BatchNormalization,
                                     Dropout, MaxPooling2D, Flatten, Dense)

def custom_loss_function(y_true, y_pred):
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)

def create_model(input_shape, config=None):
    model = Sequential()
    model.add(InputLayer(input_shape))

    for n in range(config.layers):
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(config.dropout))

    model.add(Flatten())
    model.add(Dense(config.fc_layer_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='relu'))

    return model
