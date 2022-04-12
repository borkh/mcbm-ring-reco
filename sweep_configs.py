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
                    #learning_rate =         dict(distribution="uniform",
                    #                             min=0.0002,
                    #                             max=0.002),
                    learning_rate =         dict(values=[x/10000 for x in range(2, 21, 1)]),
                    dropout =               dict(value=0.1),
                    epochs =                dict(value=50),
                    batch_size =            dict(value=32),
                    spe =                   dict(value=2000),
                    decay_steps =           dict(values=[2000, 4000, 6000, 8000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000]),
                    decay =                 dict(values=[x/100 for x in range(86, 100, 2)]),
                    # conv2D parameters
                    conv_layers =           dict(value=4),
                    nof_initial_filters =   dict(value=64),
                    conv_kernel_size =      dict(value=3),
                    padding =               dict(value="same"),
                    # max pooling parameters
                    max_pooling =           dict(value=True),
                    pool_size =             dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layers =             dict(value=1),
                    fc_layer_size =         dict(value=512),
                    fc_activation =         dict(value="relu"),
                    # shapes
                    input_shape =           dict(value=(72,32,1)),
                    output_shape =          dict(value=15),
                    min_hits_per_ring =     dict(value=24),
                    max_hits_per_ring =     dict(value=33),
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
                    epochs =                dict(value=24),
                    batch_size =            dict(value=32),
                    spe =                   dict(value=10000),
                    #decay_steps =           dict(value=30000),
                    decay =                 dict(value=0.5),
                    # conv2D parameters
                    conv_layers =           dict(value=3),
                    nof_initial_filters =   dict(value=16),
                    conv_kernel_size =      dict(value=3),
                    padding =               dict(value="same"),
                    # max pooling parameters
                    max_pooling =           dict(value=True),
                    pool_size =             dict(value=(2,2)),
                    # fully connected layer parameters
                    fc_layers =             dict(value=1),
                    fc_layer_size =         dict(value=1024),
                    fc_activation =         dict(value="relu"),
                    # shapes
                    input_shape =           dict(value=(72,32,1)),
                    output_shape =          dict(value=15),
                    min_hits_per_ring =     dict(value=24),
                    max_hits_per_ring =     dict(value=33),
                    ring_noise =            dict(value=0.08),
                    init_lr =               dict(value=1e-6),
                    max_lr =                dict(value=0.001)
    )
)
