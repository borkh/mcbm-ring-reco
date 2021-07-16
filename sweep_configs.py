#!/usr/bin/env python3
import wandb
import math

sweep_config = {
    'method': 'random',
    'metric' : {
        'name': 'loss',
        'goal': 'minimize'
        },
    'parameters': {
        'loss': {
            'values': ['MeanSquaredError']
            },

        # optimization parameters
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            },

        # convolutional layer parameters
        'conv_layers': {
            'values': [2, 3, 4, 5]
            },
        'conv_filters': {
            'values': [16, 32, 64, 128]
            },
        'conv_kernel_size': {
            'values': [tuple((2,2)), tuple((3,3))]
            },
        'kernel_initializer': {
            'values': ['glorot_normal', 'glorot_uniform']
            },
        'padding': {
            'values': ['valid', 'same']
            },

        # max_pooling parameters
        'max_pooling': {
            'values': [True, False]
            },
        'pool_size': {
            'values': [(2,2), (3,3)]
            },

        # fully connected layer parameters
        'fc_layer': {
            'values': [True, False]
            },
        'fc_layer_size': {
            'values': [128, 256, 512, 1028, 2048]
            },
        'fc_activation': {
            'values': ['relu', 'sigmoid']
            },

        'dropout': {
              'values': [0.1, 0.2, 0.3, 0.4, 0.5]
            },
        'epochs': {
            'value': 50
            },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(32),
            'max': math.log(256),
            }
    }
}
