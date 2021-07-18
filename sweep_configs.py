#!/usr/bin/env python3
import wandb
import math

sweep_config = {
    'method': 'bayes',
    'metric' : {
        'name': 'loss',
        'goal': 'minimize'
        },
    'parameters': {
        'loss': {
            'value': 'MeanSquaredError'
            },

        # optimization parameters
        'optimizer': {
            'value': ['adam', 'sgd']
            },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            },

        # convolutional layer parameters
        'conv_layers': {
            'values': [2, 3, 4, 5, 6, 7]
            },
        'conv_filters': {
            'values': [16, 32, 64, 128]
            },
        'conv_kernel_size': {
            'values': [2, 3, 4, 5]
            },
        'kernel_initializer': {
            'values': ['glorot_normal', 'glorot_uniform']
            },
        'padding': {
            'value': 'same'
            },

        # max_pooling parameters
        'max_pooling': {
            'values': [True, False]
            },
        'pool_size': {
            'values': [(2,2), (3,3)]
            },

        # fully connected layer parameters
        'fc_layer_size': {
            'values': [128, 256, 512, 1028, 2048]
            },
        'fc_activation': {
            'value': 'relu'
            },

        'dropout': {
              'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
            },
        'epochs': {
            'value': 30
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
