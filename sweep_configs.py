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
                                       max=1e-9),
                    #learning_rate=dict(values=[0.001]),
                    dropout=dict(value=0.1),
                    epochs=dict(value=5),
                    batch_size=dict(value=32),
                    spe=dict(value=200),
                    # conv2D parameters
                    conv_layers=dict(values=[4]),
                    nof_initial_filters=dict(values=[16]),
                    conv_kernel_size=dict(value=3),
                    padding=dict(value="same"),
                    # max pooling parameters
                    max_pooling=dict(value=True),
                    pool_size=dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layers=dict(value=1),
                    fc_layer_size=dict(value=256),
                    fc_activation=dict(value="relu"),
                    # shapes
                    input_shape=dict(value=(72,32,1)),
                    output_shape=dict(value=15),
    )
)

single_run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss=dict(value="MeanSquaredError"),
                    optimizer=dict(value="adam"),
                    learning_rate=dict(value=1.277e-7),
                    dropout=dict(value=0.1),
                    epochs=dict(value=5),
                    batch_size=dict(value=32),
                    spe=dict(value=5000),
                    # conv2D parameters
                    conv_layers=dict(value=4),
                    nof_initial_filters=dict(value=16),
                    conv_kernel_size=dict(value=3),
                    padding=dict(value="same"),
                    # max pooling parameters
                    max_pooling=dict(value=True),
                    pool_size=dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layers=dict(value=1),
                    fc_layer_size=dict(value=256),
                    fc_activation=dict(value="relu"),
                    # shapes
                    input_shape=dict(value=(72,32,1)),
                    output_shape=dict(value=15),
    )
)
