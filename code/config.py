# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:49:02 2021

@author: angel
"""

import tensorflow as tf

AUTO = None
REPLICAS = None
STRATEGY = None

def initialize_device(device):
    
    global AUTO, REPLICAS, STRATEGY
    
    if device == "TPU":
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            STRATEGY = tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            print('Could not connect to TPU, setting default strategy')
            tpu = None
            STRATEGY = tf.distribute.get_strategy()
            
    elif device == "GPU":
        STRATEGY = tf.distribute.MirroredStrategy()
        
    AUTO     = tf.data.experimental.AUTOTUNE
    REPLICAS = STRATEGY.num_replicas_in_sync
    print(f'Number of accelerators: {REPLICAS}')
    
    
def get_gcs_path(dataset):
    from kaggle_datasets import KaggleDatasets
    from kaggle_secrets import UserSecretsClient
    
    user_secrets = UserSecretsClient()
    user_credential = user_secrets.get_gcloud_credential()
    
    user_secrets.set_tensorflow_credential(user_credential)
    
    gcs_ds_path = KaggleDatasets().get_gcs_path(dataset)
    
    return gcs_ds_path