#!/usr/bin/env python3
import wandb
import math
from pprint import pprint

sweep_config = {
    'method': 'random',
    'metric' : {
        'name': 'loss',
        'goal': 'minimize'
        },
    'parameters': {
        'optimizer': {
            'values': ['adam']
            },
        'loss': {
            'values': ['MeanAbsoluteError']
            },
        'layers': {
            'values': [2, 3]
            },
        'fc_layer_size': {
            'values': [256, 512, 1028]
            },
        'conv_layer_size': {
            'values': [16, 32, 64]
            },
        'fc_activation': {
            'values': ['relu']
            },
        'dropout': {
              'values': [0.2, 0.3, 0.4]
            },
        'epochs': {
            'value': 15
            },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.05
            },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(16),
            'max': math.log(128),
            }
    }
}
