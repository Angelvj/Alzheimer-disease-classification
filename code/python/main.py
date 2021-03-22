# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:42:09 2021

@author: angel
"""

import numpy as np
import tensorflow as tf
import keras as k
import nibabel as nib

import glob
import os
# import tqdm


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
    
    for filename in glob.iglob(dirname + '/*.nii'):
        imgs.append(load_image(filename))
        
    imgs = np.stack(imgs)
    return imgs


# Probamos a cargar un directorio
dirname = 'ppNOR/MRI/whiteMatter'
imgs = load_images_from_dir(DATA_PATH + '/' + dirname)


# with open('/content/drive/My Drive/Machine learning/foo.txt', 'w') as f:
#   f.write('Hello Google Drive!')
# !cat /content/drive/My\ Drive/Machine\ learning/foo.txt