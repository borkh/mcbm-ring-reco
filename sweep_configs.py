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
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss =                  dict(value="mse"),
                    # learning rate
                    max_lr =                dict(value=0.05),
                    init_lr =               dict(value=1e-6),
                    lr_decay =              dict(value=0.8),
                    decay_length =          dict(values=[2,3,4]),

                    # regularization
                    fc_dropout =            dict(value=0.0),
                    output_dropout =        dict(value=0.0),

                    # epochs
                    epochs =                dict(value=24),
                    batch_size =            dict(values=[100,200,300]),

                    # conv2D parameters
                    conv_layers =           dict(values=[3,4,5]),
                    nof_initial_filters =   dict(values=[16,32,64]),
                    conv_kernel_size =      dict(values=[3,5]),

                    # fully connected layer parameters
                    fc_layer_size =         dict(values=[64,128,256]),
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
