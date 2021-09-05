# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 10:08:36 2021

@author: angel
"""

def change_input_shape(model, new_shape, name=None):
    new_shape = [None] + list(new_shape)
    new_shape = tuple(new_shape)
    # Extract model's configuration
    model_config = model.get_config()
    # Change config
    if name is not None:
        input_layer_name = name
    else:
        input_layer_name = model_config['layers'][0]['name']
    model_config['layers'][0] = {
                        'name': input_layer_name,
                        'class_name': 'InputLayer',
                        'config': {
                            'batch_input_shape': new_shape,
                            'dtype': 'float32',
                            'sparse': False,
                            'name': input_layer_name
                        },
                        'inbound_nodes': []
                    }
    model_config['layers'][1]['inbound_nodes'] = [[[input_layer_name, 0, 0, {}]]]
    model_config['input_layers'] = [[input_layer_name, 0, 0]] 
    # Create new model
    new_model = model.__class__.from_config(model_config, custom_objects={})
    # Copy weights
    weights = [layer.get_weights() for layer in model.layers[1:]]
    for layer, weight in zip(new_model.layers[1:], weights):
        layer.set_weights(weight)

    return new_model