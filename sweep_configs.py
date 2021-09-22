#!/usr/bin/env python3
import wandb
import math
from pprint import pprint

sweep_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss=dict(value="MeanSquaredError"),
                    optimizer=dict(value="adam"),
                    learning_rate=dict(distribution="uniform",
                                       min=0,
                                       max=0.001),
                    dropout=dict(values=[0.3, 0.4]),
                    epochs=dict(value=50),
                    # conv2D parameters
                    conv_layers=dict(values=[3, 4, 5]),
                    nof_initial_filters=dict(values=[8, 16, 32]),
                    conv_kernel_size=dict(values=[7, 9]),

                    kernel_initializer=dict(value="glorot_normal"),
                    padding=dict(value="same"),
                    # max pooling parameters
                    max_pooling=dict(value=True),
                    pool_size=dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layer_size=dict(value=512),
                    fc_activation=dict(value="relu"),
    )
)

single_run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss=dict(value="MeanSquaredError"),
                    optimizer=dict(value="adam"),
                    learning_rate=dict(value=0.0005792),
                    dropout=dict(value=0.3),
                    epochs=dict(value=20),
#                    batch_size=dict(value=45),
                    # conv2D parameters
                    conv_layers=dict(value=3),
                    nof_initial_filters=dict(value=13),
                    conv_kernel_size=dict(value=5),
                    kernel_initializer=dict(value="glorot_normal"),
                    padding=dict(value="same"),
                    # max pooling parameters
                    max_pooling=dict(value=True),
                    pool_size=dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layer_size=dict(value=512),
                    fc_activation=dict(value="relu"),
    )
)
