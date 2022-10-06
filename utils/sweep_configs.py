run_config = dict(
    method="random",
    metric=dict(name='loss',
                goal='minimize'
    ),
    parameters=dict(loss =                  dict(value="mse"),
                    # learning rate
                    max_lr =                dict(value=0.1),
                    init_lr =               dict(value=0.004),

                    # lr schedule
                    mom_min =               dict(value=0.85),
                    mom_max =               dict(value=0.95),
                    phase0perc =            dict(value=0.3),

                    # epochs
                    epochs =                dict(value=10),
                    batch_size =            dict(value=256),

                    # conv2D parameters
                    conv_layers =           dict(value=5),
                    nof_initial_filters =   dict(value=64),
                    conv_kernel_size =      dict(value=3),

                    # fully connected layer parameters
                    fc_layer_size =         dict(value=512),
                    fc_activation =         dict(value="relu"),
    )
)
