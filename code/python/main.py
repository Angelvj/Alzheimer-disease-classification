# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:42:09 2021

@author: angel
"""

import numpy as np
import tensorflow as tf
import keras as k
import nibabel as nib
from sklearn.model_selection import train_test_split

import time
import glob
from tqdm import tqdm


COLAB = False   # Select between colaboratory or local execution

if COLAB:
  from google.colab import drive
  drive.mount('/content/drive')
  DATA_PATH = '/content/drive/My Drive/Machine learning/data'

else: 
  DATA_PATH = '../../data'


def load_image(filename):    
    """
    
    Parameters
    ----------
    filename : str
        relative path to de image

    Returns
    -------
    img : numpy ndarray
        array containing the image
        
    """
    img = nib.load(filename)
    img = np.asarray(img.dataobj)
    return img
    

def load_images_from_dir(dirname):
    """
    
    Parameters
    ----------
    dirname : str
        name of the directory containing images.

    Returns
    -------
    imgs : numpy ndarray
        array containing all of the images in the folder.

    """
    imgs = []
    
    for filename in tqdm(glob.glob(dirname + '/*.nii')):
        imgs.append(load_image(filename))
        
    imgs = np.stack(imgs)
    return imgs


def load_data(dirs_dict):
    """
    
    Parameters
    ----------
    dirs_dict : dictionary
        dictionary containing data folders name, and the label for the images
        on each forlder.

    Returns
    -------
    x : numpy ndarray
        array containing the images.
    y : numpy ndarray
        array containig the label of each image.

    """
    first = True
    for key, value in dirs_dict.items():
        if first:
            x = load_images_from_dir(value)
            y = np.full((x.shape[0]), key, dtype=np.uint8)
            first = False
        else:
            x_current = load_images_from_dir(value)
            x = np.concatenate((x, x_current))
            y = np.concatenate((y, np.full((x_current.shape[0]), key, dtype=np.uint8)))
            
    y = k.utils.to_categorical(y)
    
    return x, y


# Load PET images with labels
print('\n --- Loading PET data --- \n')
time.sleep(0.5)
X, y = load_data({0: DATA_PATH + "/ppNOR/PET", 
                  1: DATA_PATH + "/ppAD/PET",
                  2: DATA_PATH + "/ppMCI/PET"})


# Separate into training and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, stratify = y, random_state = 1)