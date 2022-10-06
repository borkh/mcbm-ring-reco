#!/usr/bin/env python3
import numpy as np, tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D, Cropping2D, Flatten, Dense, Reshape, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from utils.utils import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#import matplotlib
#matplotlib.use("TkAgg")

def build_autoencoder(input_shape):
    inputs = Input(input_shape)

    # encoder
    x = BatchNormalization()(inputs)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    encoded = MaxPooling2D(2, padding='same')(x)

    # regressor
    x = Flatten()(encoded)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(25, activation='relu')(x)
    regressor = Reshape((5,5))(x)

    # decoder
    x = BatchNormalization()(encoded)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)

    model = Model(inputs, [regressor, decoded])
    model.summary()

    return model

def train(x, y):
    # add noise to data
    x_noisy = x + 0.21 * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    x_noisy = np.round(np.clip(x_noisy, 0., 1.), 0)

    ae = build_autoencoder(x.shape[1:])
    ae.compile(optimizer='adam', loss=['mse', 'binary_crossentropy'], metrics=['acc'])

    ae.fit(x_noisy, [y, x],
           epochs=50,
           batch_size=128,
           shuffle=True,
           validation_split=0.1,
           callbacks=tf.keras.callbacks.ModelCheckpoint('models/autoencoder_regressor_mcbm.model', monitor='val_loss'))

def plots(x_test, x_pred):
    fig, ax = plt.subplots(2,5)
    for i in range(5):
        ax[0,i].imshow(x_test[i], cmap='gray')

    for i in range(5):
        ax[1,i].imshow(x_pred[i], cmap='gray')
    plt.show()

# load x and y from file data/10k_no_noise.npz
data = np.load('data/10k_no_noise.npz')
x, y = data['arr_0'], data['arr_1']
# split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# add gaussian noise to x_test with values of 1.0
x_test_noisy = x_test + 0.21 * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.round(np.clip(x_test_noisy, 0., 1.), 0)


#%%
# train the autoencoder
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#train(x_train, y_train)


# predictions
x_test, y_test = np.load('data/10k.npz')['arr_0'], np.load('data/10k.npz')['arr_1']

ae = tf.keras.models.load_model('models/autoencoder_regressor_mcbm.model')
y_pred, x_pred = ae.predict(x_test)
#x_pred = np.round(x_pred, decimals=0)


x_pred = np.array([cv2.merge((a,a,a)) for a in x_pred])

cnn_fit = np.array([plot_single_event(x, y) for x,y in zip(x_pred[:100], y_test[:100])])

for i in range(10):
    plots(x_test[i*5:i*5+5], cnn_fit[i*5:i*5+5])
