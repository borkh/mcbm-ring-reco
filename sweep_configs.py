#!/usr/bin/env python3
import wandb
import math
from pprint import pprint
import random as rand
import numpy as np

sweep_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss=dict(value="mse"),
    )
)

single_run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss =                  dict(value="mse"),
                    # learning rate
                    max_lr =                dict(value=0.001),
                    init_lr =               dict(value=1e-6),
                    lr_decay =              dict(value=0.5),

                    # epochs
                    epochs =                dict(value=1000),
                    batch_size =            dict(value=32),
                    spe =                   dict(value=4000),

                    # conv2D parameters
                    conv_layers =           dict(value=3),
                    nof_initial_filters =   dict(value=16),
                    conv_kernel_size =      dict(value=3),
                    padding =               dict(value="same"),
                    conv_dropout =          dict(value=0.1),

                    # max pooling parameters
                    pool_size =             dict(value=(2,2)),
                    fc_layers =             dict(value=1),

                    # fully connected layer parameters
                    fc_layer_size =         dict(value=1024),
                    fc_activation =         dict(value="relu"),
                    fc_dropout =            dict(value=0.5),

                    # shapes
                    input_shape =           dict(value=(72,32,1)),
                    output_shape =          dict(value=15),

                    # ring parameters
                    min_hits_per_ring =     dict(value=24),
                    max_hits_per_ring =     dict(value=33),
                    ring_noise =            dict(value=0.08)
    )
)
