import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,  # type: ignore
                                     Flatten, Input, MaxPooling2D, Reshape)
from tensorflow.keras.models import Model  # type: ignore

# create wandb configuration dictionary which contains all the hyperparameters
# that will be used in train.py and in build_model()
run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
                ),
    parameters=dict(loss=dict(value="mse"),
                    # learning rate
                    max_lr=dict(value=0.05),
                    init_lr=dict(value=1e-3),

                    # epochs
                    epochs=dict(value=5),
                    batch_size=dict(value=200),

                    # conv2D parameters
                    conv_layers=dict(value=5),
                    nof_initial_filters=dict(value=32),
                    conv_kernel_size=dict(value=3),
                    )
)


def build_model(input_shape, config):
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
    output = Reshape((5, -1))(t)

    model = Model(input_, output)
    model.summary()
    return model


def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)
