#!/usr/bin/env python3
import wandb
import math
from pprint import pprint
import random as rand
import numpy as np

run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss =                  dict(value="mse"),
                    # learning rate
                    max_lr =                dict(value=0.08),
                    init_lr =               dict(value=0.008),

                    # lr schedule
                    mom_min =               dict(value=0.80),
                    mom_max =               dict(value=0.95),
                    phase0perc =            dict(value=0.3),

                    # epochs
                    epochs =                dict(value=6),
                    batch_size =            dict(value=256),

                    # conv2D parameters
                    conv_layers =           dict(value=5),
                    nof_initial_filters =   dict(value=64),
                    conv_kernel_size =      dict(value=3),

                    # fully connected layer parameters
                    fc_layer_size =         dict(value=64),
                    fc_activation =         dict(value="relu"),

                    # shapes
                    input_shape =           dict(value=(72,32,1)),
                    output_shape =          dict(value=(5,5)),

                    # ring parameters
                    min_hits_per_ring =     dict(value=12),
                    max_hits_per_ring =     dict(value=25),
                    ring_noise =            dict(value=0.08),
                    spe =                   dict(value=1000),
    )
)

sweep_config = dict(
    method="grid",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss =                  dict(value="mse"),
                    # learning rate
                    max_lr =                dict(value=0.2),
                    init_lr =               dict(value=1e-5),
                    lr_decay =              dict(value=0.8),
                    decay_length =          dict(value=4),

                    # regularization
                    fc_dropout =            dict(value=0.0),

                    # lr schedule
                    mom_min =               dict(values=[0.9, 0.85, 0.75, 0.65, 0.60]),
                    mom_max =               dict(values=[0.98, 0.95, 0.91]),

                    # epochs
                    epochs =                dict(values=[5,10,15,20]),
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
