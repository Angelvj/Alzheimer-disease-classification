# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:42:09 2021

@author: angel
"""

import numpy as np
import tensorflow as tf
import keras as k
import nibabel as nib



COLAB = False

if COLAB:
  from google.colab import drive
  drive.mount('/content/drive')
  DATA_PATH = '/content/drive/My Drive/Machine learning/data'

else: 
  DATA_PATH = '../../data'

# Prueba de cómo cargar una imagen con Nibabel

filename_gray_matter = DATA_PATH + '/Datos_TFG_Angel/ppNOR/MRI/m0wrp2ADNI_941_S_1203_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070801201705724_S25671_I63879.nii'
filename_white_matter = DATA_PATH + '/Datos_TFG_Angel/ppNOR/MRI/m0wrp1ADNI_941_S_1203_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070801201705724_S25671_I63879.nii'

gray_matter = nib.load(filename_gray_matter)
white_matter = nib.load(filename_white_matter)

gray_image_data = np.array(gray_matter.get_fdata())
white_image_data = np.array(white_matter.get_fdata())

print(np.sum(np.isnan(gray_image_data)))
print(np.sum(np.isnan(white_image_data)))

# Todo el borde de la imagen tiene valor NaN
shape = gray_image_data.shape
print(121*145*2 + 119*145*2 + 119*119*2)


# print(shape[0]*shape[1] + shape[0]*)

print(np.mean(gray_image_data[~np.isnan(gray_image_data)]))
print(np.mean(white_image_data[~np.isnan(white_image_data)]))

# with open('/content/drive/My Drive/Machine learning/foo.txt', 'w') as f:
#   f.write('Hello Google Drive!')
# !cat /content/drive/My\ Drive/Machine\ learning/foo.txt