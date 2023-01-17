import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,  # type: ignore
                                     Flatten, Input, MaxPooling2D, Reshape)
from tensorflow.keras.models import Model  # type: ignore
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

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
                    init_lr=dict(value=0.04),

                    # epochs
                    epochs=dict(value=10),
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


# def hits_on_ring(y_true, y_pred):
#     pars = y_pred[..., :5]
#     hits = tf.reshape(y_true[..., 5:],
#                       (y_pred.shape[0], y_pred.shape[1], -1, 2))

#     # initialize two tensors to store the number of hits per ring
#     # per ring per event in nof_hits and the penalty in penalty
#     nof_hits = tf.zeros((y_pred.shape[0], y_pred.shape[1]), tf.float64)
#     penalty = tf.zeros((y_pred.shape[0], y_pred.shape[1]), tf.float64)
#     # loop over all events, rings and hits
#     for i in tf.range(y_pred.shape[0]):
#         for j in tf.range(y_pred.shape[1]):
#             for k in tf.range((y_pred.shape[2]-5)//2):
#                 hits_ = hits[i, j, k]
#                 pars_ = pars[i, j]

#                 equal_zero = tf.reduce_all(
#                     tf.equal(hits_, tf.constant(0., dtype=tf.float64)))
#                 if not equal_zero:
#                     x = hits_[1] + 0.5 - pars_[0]
#                     y = hits_[0] + 0.5 - pars_[1]

#                     distance = tf.sqrt(x**2 + y**2)
#                     # is_close: tf.Tensor = tf.constant(0., dtype=tf.double)
#                     # if tf.math.abs(distance - pars_[2]) < 1.5:
#                     #     is_close = tf.constant(1., dtype=tf.double)
#                     is_close = tf.math.abs(distance - pars_[2]) < 1.5
#                     is_close = tf.cast(is_close, tf.float64)

#                     a, b = i.numpy(), j.numpy()
#                     indices = tf.constant([[a, b]], tf.int64)
#                     updates = tf.constant([is_close.numpy()], tf.float64)
#                     nof_hits = tf.tensor_scatter_nd_add(
#                         nof_hits, indices, updates)

#     for i in tf.range(y_pred.shape[0]):
#         for j in tf.range(y_pred.shape[1]):
#             less_than_10 = tf.less(nof_hits[i, j], 4)
#             all_zero = tf.reduce_all(
#                 tf.equal(y_true[i, j, :5], tf.constant(0., dtype=tf.float64)), axis=0)

#             if not all_zero and less_than_10:
#                 tf.print('penalty added')
#                 a, b = i.numpy(), j.numpy()
#                 indices = tf.constant([[a, b]], tf.int64)
#                 updates = tf.constant([1.], tf.float64)
#                 penalty = tf.tensor_scatter_nd_add(penalty, indices, updates)
#             else:
#                 tf.print('no penalty added')

#     return tf.reduce_sum(penalty)

def hits_on_ring(y_true, y_pred):
    pars = y_pred[..., :5]
    hits = y_true[..., 5:]
    hits = hits.reshape((y_pred.shape[0], y_pred.shape[1], -1, 2))

    # check the number of hits on the ring for each ring in each event
    # and store it in nof_hits
    nof_hits = np.zeros((y_pred.shape[0], y_pred.shape[1]))
    for i, j, k in np.ndindex((y_pred.shape[0], y_pred.shape[1], (y_pred.shape[2]-5) // 2)):
        hits_ = hits[i, j, k]
        pars_ = pars[i, j]
        if np.all(hits_.numpy() != 0.):
            # equal_zero = tf.reduce_all(tf.equal(hits_, tf.constant(0.)))
            # if not equal_zero:
            x = hits_[1] + 0.5 - pars_[0]
            y = hits_[0] + 0.5 - pars_[1]
            if np.isclose(np.sqrt(x**2 + y**2), pars_[2], atol=1.5):
                nof_hits[i, j] += 1

    # if there are less than 10 hits on a ring with ring parameters
    # that are not all zero, add a penalty of 1 to the loss
    penalty = np.zeros((y_pred.shape[0], y_pred.shape[1]))
    for i, j in np.ndindex((y_pred.shape[0], y_pred.shape[1])):
        if nof_hits[i, j] < 7 and not np.all(y_true[i, j, :5] == 0):
            penalty[i, j] = 1

    return np.sum(penalty)


def custom_loss(y_true, y_pred):
    # penalty = hits_on_ring(y_true, y_pred)

    # y_true = y_true[..., :5]
    # y_pred = y_pred[..., :5]

    return tf.keras.losses.mean_squared_error(y_true, y_pred)  # + penalty


if __name__ == '__main__':
    import tensorflow as tf  # nopep8
    import sys  # nopep8
    sys.path.append('..')
    from data.create_data import DataGen  # nopep8
    from utils.utils import display_images, fit_rings  # nopep8
    np.set_printoptions(threshold=sys.maxsize)

    dg = DataGen('../data/train', batch_size=32)
    X, y = dg[0]

    model_path = 'checkpoints/' + '0M-202301122112.model'
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'custom_loss': custom_loss})
    pred = model.predict(X)  # type: ignore
    # fit_rings(X, pred)

    # convert y and pred to tensor
    y = tf.convert_to_tensor(y)
    pred = tf.convert_to_tensor(pred)

    loss = tf.reduce_mean(custom_loss(y, pred))
    print(loss)
