#!/usr/bin/env python3
import wandb
import math
from pprint import pprint
import random as rand
import numpy as np

single_run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss =                  dict(value="mse"),
                    # learning rate
                    max_lr =                dict(value=0.05),
                    init_lr =               dict(value=1e-6),
                    lr_decay =              dict(value=0.8),
                    decay_length =          dict(value=4),

                    # regularization
                    fc_dropout =            dict(value=0.0),

                    # epochs
                    epochs =                dict(value=40),
                    batch_size =            dict(value=256),

                    # conv2D parameters
                    conv_layers =           dict(value=4),
                    nof_initial_filters =   dict(value=64),
                    conv_kernel_size =      dict(value=3),

                    # fully connected layer parameters
                    fc_layer_size =         dict(value=64),
                    fc_activation =         dict(value="relu"),

                    # shapes
                    input_shape =           dict(value=(72,32,1)),
                    output_shape =          dict(value=15),

                    # ring parameters
                    min_hits_per_ring =     dict(value=12),
                    max_hits_per_ring =     dict(value=21),
                    ring_noise =            dict(value=0.08)
    )
)

sweep_config = dict(
    method="grid",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss =                  dict(value="mse"),
                    # learning rate
                    max_lr =                dict(value=0.05),
                    init_lr =               dict(values=[1e-6,2e-6,4e-6,6e-6,8e-6,1e-5,2e-5,4e-5,6e-5,8e-5,1e-4,2e-4,4e-5,6e-4,8e-4]),
                    lr_decay =              dict(value=0.8),
                    decay_length =          dict(value=4),

                    # regularization
                    fc_dropout =            dict(value=0.0),

                    # epochs
                    epochs =                dict(value=48),
                    batch_size =            dict(value=256),

                    # conv2D parameters
                    conv_layers =           dict(value=4),
                    nof_initial_filters =   dict(value=64),
                    conv_kernel_size =      dict(value=3),

                    # fully connected layer parameters
                    fc_layer_size =         dict(value=64),
                    fc_activation =         dict(value="relu"),

                    # shapes
                    input_shape =           dict(value=(72,32,1)),
                    output_shape =          dict(value=15),

                    # ring parameters
                    min_hits_per_ring =     dict(value=12),
                    max_hits_per_ring =     dict(value=21),
                    ring_noise =            dict(value=0.08)
    )
)
