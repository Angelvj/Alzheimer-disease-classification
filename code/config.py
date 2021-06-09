# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:49:02 2021

@author: angel
"""

import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = None
STRATEGY = None
TPU_CR = None

def initialize_device(device):
    
    global AUTO, REPLICAS, STRATEGY, TPU_CR
    
    if device == "TPU":
        try:
            TPU_CR = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(TPU_CR)
            tf.tpu.experimental.initialize_tpu_system(TPU_CR)
            STRATEGY = tf.distribute.experimental.TPUStrategy(TPU_CR)
        except ValueError:
            print('Could not connect to TPU, setting default strategy')
            TPU_CR = None
            STRATEGY = tf.distribute.get_strategy()
            
    elif device == "GPU":
        STRATEGY = tf.distribute.MirroredStrategy()
        
    REPLICAS = STRATEGY.num_replicas_in_sync
    print(f'Number of accelerators: {REPLICAS}')