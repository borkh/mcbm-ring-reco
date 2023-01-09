import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,  # type: ignore
                                     Flatten, Input, MaxPooling2D, Reshape)
from tensorflow.keras.models import Model  # type: ignore
import numpy as np
import sys

# create wandb configuration dictionary which contains all the hyperparameters
# that will be used in train.py and in build_model()
run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
                ),
    parameters=dict(loss=dict(value="mse"),
                    # learning rate
                    max_lr=dict(value=0.1),
                    init_lr=dict(value=0.001),

                    # epochs
                    epochs=dict(value=5),
                    batch_size=dict(value=100),

                    # conv2D parameters
                    conv_layers=dict(value=5),
                    nof_initial_filters=dict(value=16),
                    conv_kernel_size=dict(value=3),
                    )
)


def build_model(input_shape, config):
    nif = config.nof_initial_filters
    bmomentum = 0.95
    nof_conv_layers_per_block = [2, 2, 3, 3, 3]

    input_ = Input(input_shape)
    t = input_

    for i, nof_conv_layers in enumerate(nof_conv_layers_per_block):
        for _ in range(nof_conv_layers):
            t = BatchNormalization(momentum=bmomentum)(t)
            t = Conv2D(filters=config.nof_initial_filters * 2**i,
                       kernel_size=config.conv_kernel_size,
                       padding="same",
                       activation="relu",
                       kernel_initializer="he_uniform")(t)
        t = BatchNormalization(momentum=bmomentum)(t)
        t = MaxPooling2D((2, 2), padding="same")(t)

    t = Flatten()(t)
    t = BatchNormalization(momentum=bmomentum)(t)
    t = Dense(25, kernel_initializer="he_uniform", activation="relu",
              name="predictions")(t)
    output = Reshape((5, 5))(t)

    model = Model(input_, output)
    model.summary()
    return model


i = 0


def custom_loss(train_generator):

    def mse(y_true, y_pred):
        global i
        X = next(train_generator)[0]

        nof_hits = np.where(X == 1)[0].shape[0]
        penalty = 1e6 if nof_hits < 50 else 0

        tf.print(f'\n X.shape: {X.shape}')
        tf.print(f'\n nof_hits: {nof_hits}')
        tf.print(f'\n i: {i}')
        i += 1

        return tf.keras.losses.mean_squared_error(y_true, y_pred) + penalty

    return mse

    # if not hits_near_rings(y_true, y_pred):
    #     mse += 1e6
