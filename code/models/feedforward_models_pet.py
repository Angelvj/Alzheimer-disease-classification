# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 11:25:13 2021

@author: angel
"""

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling3D, Flatten, Dense, Conv3D, BatchNormalization, Input, Dropout
from tensorflow.keras.regularizers import l2


def model_0(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=5, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)
    
    x = Flatten()(x)
    x = Dense(units=256, activation="relu")(x)
    
    outputs = Dense(units=3, activation="softmax")(x)
   
    model = tf.keras.Model(inputs, outputs, name="model_0_pet")
    return model


def model_1(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    
    x = Conv3D(filters=32, kernel_size=5, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)
    
    x = Conv3D(filters=32, kernel_size=5, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    
    x = Flatten()(x)
    x = Dense(units=256, activation="relu")(x)
    outputs = Dense(units=3, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="model_1_pet")
    return model


def model_2(input_shape):
    
    inputs = tf.keras.layers.Input(input_shape)

    x = Conv3D(filters=16, kernel_size=5, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=64, kernel_size=5, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=128, kernel_size=5, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(units=256, activation="relu")(x)

    outputs = Dense(units=3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_2_pet")
    return model


def model_3(input_shape):
    
    inputs = tf.keras.layers.Input(input_shape)

    x = Conv3D(filters=16, kernel_size=3, activation='relu')(inputs)
    x = Conv3D(filters=16, kernel_size=3, activation='relu')(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=64, kernel_size=3, activation='relu')(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(units=256, activation="relu")(x)

    outputs = Dense(units=3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_3_pet")
    return model


def model_4(input_shape):

    inputs = Input(input_shape)

    x = Conv3D(filters=16, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(inputs)
    x = Conv3D(filters=16, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = Conv3D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = Conv3D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = BatchNormalization(momentum=0.9)(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling3D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=128, activation='relu')(x)

    outputs = Dense(units=3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_4_pet")
    return model


def model_4_augmentation(input_shape):

    inputs = Input(input_shape)

    x = Conv3D(filters=16, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(inputs)
    x = Conv3D(filters=16, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = Conv3D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = Conv3D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = BatchNormalization(momentum=0.95)(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling3D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=128, activation='relu')(x)

    outputs = Dense(units=3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_4_pet")
    return model