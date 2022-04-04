#!/usr/bin/env python3
import wandb
import math
from pprint import pprint
import random as rand

sweep_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss=dict(value="mse"),
                    learning_rate =         dict(value=0.001),
                    dropout =               dict(value=0.1),
                    epochs =                dict(value=100),
                    batch_size =            dict(value=32),
                    spe =                   dict(value=2000),
                    # conv2D parameters
                    conv_layers =           dict(values=[3, 4, 5]),
                    nof_initial_filters =   dict(values=[8, 16, 32]),
                    conv_kernel_size =      dict(values=[3, 5, 7]),
                    padding =               dict(value="same"),
                    # max pooling parameters
                    max_pooling =           dict(value=True),
                    pool_size =             dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layers =             dict(value=1),
                    fc_layer_size =         dict(values=[256, 512, 1024, 2048]),
                    fc_activation =         dict(value="relu"),
                    # shapes
                    input_shape =           dict(value=(72,32,1)),
                    output_shape =          dict(value=15),
                    hits_per_ring =         dict(value=rand.randint(24, 44)),
                    ring_noise =            dict(value=0.08)
    )
)

single_run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss =                  dict(value="mse"),
                    learning_rate =         dict(value=0.001),
                    dropout =               dict(value=0.1),
                    epochs =                dict(value=100),
                    batch_size =            dict(value=32),
                    spe =                   dict(value=2000),
                    # conv2D parameters
                    conv_layers =           dict(value=4),
                    nof_initial_filters =   dict(value=16),
                    conv_kernel_size =      dict(value=3),
                    padding =               dict(value="same"),
                    # max pooling parameters
                    max_pooling =           dict(value=True),
                    pool_size =             dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layers =             dict(value=1),
                    fc_layer_size =         dict(value=256),
                    fc_activation =         dict(value="relu"),
                    # shapes
                    input_shape =           dict(value=(72,32,1)),
                    output_shape =          dict(value=15),
                    hits_per_ring =         dict(value=rand.randint(24, 44)),
                    ring_noise =            dict(value=0.08)
    )
)
