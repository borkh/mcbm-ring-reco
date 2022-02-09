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
                    epochs=dict(value=35),
                    # conv2D parameters
                    conv_layers=dict(values=[3, 4, 5]),
                    nof_initial_filters=dict(values=[2, 4, 8]),
                    conv_kernel_size=dict(values=[5, 7, 9]),

                    kernel_initializer=dict(value="glorot_normal"),
                    padding=dict(value="same"),
                    # max pooling parameters
                    max_pooling=dict(value=True),
                    pool_size=dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layer_size=dict(value=128),
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
                    learning_rate=dict(value=0.000027),
                    dropout=dict(value=0.15),
                    epochs=dict(value=15),
#                    batch_size=dict(value=45),
                    # conv2D parameters
                    conv_layers=dict(value=4),
                    nof_initial_filters=dict(value=8),
                    conv_kernel_size=dict(value=3),
                    kernel_initializer=dict(value="glorot_normal"),
                    padding=dict(value="same"),
                    # max pooling parameters
                    max_pooling=dict(value=True),
                    pool_size=dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layer_size=dict(value=256),
                    fc_activation=dict(value="relu"),
    )
)
