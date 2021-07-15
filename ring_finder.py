#!/usr/bin/env python3
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os
from cbm_functions import *


tf.keras.backend.clear_session()
"""
# in case of errors
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""


## loading data
data_dir = "./datasets/"
displays_scaled = np.load(data_dir + "displays_scaled.npy", "r")
rings = np.load(data_dir + "rings.npy", "r")

def custom_loss_function(y_true, y_pred):
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)

input_shape = displays_scaled[0].shape

model = Sequential()
model.add(InputLayer(input_shape))

model.add(Conv2D(64, (3,3), padding='same', activation='relu',
                 kernel_initializer='glorot_normal', kernel_regularizer=None))
model.add(BatchNormalization(momentum=0.99))

model.add(Conv2D(64, (3,3), padding='same', activation='relu',
                 kernel_initializer='glorot_normal', kernel_regularizer=None))
model.add(BatchNormalization(momentum=0.99))

model.add(Conv2D(64, (3,3), padding='same', activation='relu',
                 kernel_initializer='glorot_normal', kernel_regularizer=None))
model.add(BatchNormalization(momentum=0.99))

model.add(Conv2D(1, (3,3), padding='same', activation='relu',
                 kernel_initializer='glorot_normal',
                 kernel_regularizer=None))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=custom_loss_function, metrics='accuracy')
model.fit(displays_scaled, rings, batch_size=100, epochs=20, validation_split=0.3)

timestr = time.strftime("%H%M")
model.save("models/ring-conv-{}.model".format(timestr))

os.system("notify-send 'training done' -u critical")
