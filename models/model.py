#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Flatten, Reshape,
                                     Dense, Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts
from keras_lr_finder import LRFinder

def build_model(input_shape, output_shape, config=None):
    input_ = Input(input_shape)
    t = input_

    t = BatchNormalization(momentum=0.95)(t)
    for n in range(config.conv_layers):
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)), # double the number of filters in each subsequent layer
                   config.conv_kernel_size,
                   kernel_initializer="he_uniform",
                   padding="same",
                   activation="relu")(t)
        t = BatchNormalization(momentum=0.95)(t)
        t = Conv2D(int(2 ** (np.log2(config.nof_initial_filters) + n)), # double the number of filters in each subsequent layer
                   config.conv_kernel_size,
                   kernel_initializer="he_uniform",
                   padding="same",
                   activation="relu")(t)
        t = BatchNormalization(momentum=0.95)(t)
        t = MaxPooling2D((2,2), padding="same")(t)

    t = Flatten()(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Dense(config.fc_layer_size, kernel_initializer="he_uniform", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Dense(25, kernel_initializer="he_uniform", activation="relu",
            name="predictions")(t)
    output = Reshape((5,5))(t)

    model = Model(input_, output)
    model.summary()
    return model

def build_model2(input_shape, output_shape, config=None):
    input_ = Input(input_shape)
    t = input_

    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(64, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(64, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = MaxPooling2D((2,2), padding="same")(t)

    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(128, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(128, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = MaxPooling2D((2,2), padding="same")(t)

    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(256, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(256, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(256, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = MaxPooling2D((2,2), padding="same")(t)

    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(512, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(512, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(512, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = MaxPooling2D((2,2), padding="same")(t)

    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(1024, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(1024, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Conv2D(1024, config.conv_kernel_size, kernel_initializer="he_uniform", padding="same", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = MaxPooling2D((2,2), padding="same")(t)

    t = Flatten()(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Dense(config.fc_layer_size, kernel_initializer="he_uniform", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Dense(25, kernel_initializer="he_uniform", activation="relu",
            name="predictions")(t)
    output = Reshape((5,5))(t)

    model = Model(input_, output)
    model.summary()
    return model

def build_efficient_net(input_shape, output_shape, config=None):
    efficient_net = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape)
    #efficient_net = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape, weights=None)

#    for layer in efficient_net.layers:
#        layer.trainable = False

    #t = Flatten()(efficient_net.get_layer('block4_conv3').output)
    t = Flatten()(efficient_net.output)
    t = BatchNormalization(momentum=0.95)(t)
    t = Dense(512, kernel_initializer="he_uniform", activation="relu")(t)
    t = BatchNormalization(momentum=0.95)(t)
    t = Dense(25, kernel_initializer="he_uniform", activation="relu", name="predictions")(t)
    output = Reshape((5,5))(t)

    model = Model(efficient_net.input, output)
    model.summary()
    return model

def build_ae(input_shape):
    input_ = Input(input_shape)

    t = Conv2D(256, 3, kernel_initializer='he_normal', padding='same', activation='relu')(input_)
    t = MaxPooling2D(2, padding='same')(t)
    t = Conv2D(128, 3, kernel_initializer='he_normal', padding='same', activation='relu')(t)
    t = MaxPooling2D(2, padding='same')(t)
    t = Conv2D(64, 3, kernel_initializer='he_normal', padding='same', activation='relu')(t)
    t = MaxPooling2D(2, padding='same')(t)

    t = Conv2D(64, 3, kernel_initializer='he_normal', padding='same', activation='relu')(t)
    t = UpSampling2D(2)(t)
    t = Conv2D(128, 3, kernel_initializer='he_normal', padding='same', activation='relu')(t)
    t = UpSampling2D(2)(t)
    t = Conv2D(256, 3, kernel_initializer='he_normal', padding='same', activation='relu')(t)
    t = UpSampling2D(2)(t)
    output = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='relu')(t)

    model = Model(input_, output)
    model.summary()

    return model

def find_lr_range(x, y):
    model = build_efficient_net(x.shape[1:], y.shape[-1])

    lr = 0.001
    opt= SGD(lr)
    model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])

    lr_finder = LRFinder(model)
    lr_finder.find(x, y, start_lr=1e-7, end_lr=5, batch_size=100, epochs=5)
    lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
    plt.show()

if __name__ == "__main__":
#    from create_data import *
#    with open("data/100k-final.pkl", "rb") as f:
#        x, y = pkl.load(f)
#        x = np.array([cv2.merge((a,a,a)) for a in x])
#    find_lr_range(x, y)
#    build_efficient_net((72,32,3), (5,5))
    build_ae((72,32,1))
