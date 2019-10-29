#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:12:50 2019

@author: uqdlund
"""

import numpy as np
from keras import backend as K
import nibabel as nib

y_pred = nib.load('/scratch/cai/DEEPSEACAT/data/20191023_nepoch60_patch32_nfilt64_batch32_dilated_lab1_8/prediction/validation_case_61/prediction.nii.gz')

y_true = nib.load('/scratch/cai/DEEPSEACAT/data/20191023_nepoch60_patch32_nfilt64_batch32_dilated_lab1_8/prediction/validation_case_61/truth.nii.gz')

labels = (1, 2, 3, 4, 5, 6)


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], data.shape[1], data.shape[2], n_labels]
    y = np.zeros(new_shape, np.int32)    
    
    for label_index in range(n_labels):
        if labels is not None:
            y[:,:,:, label_index][data == labels[label_index]] = 1
        else:
            y[:,:,:, label_index][data == (label_index + 1)] = 1
    return y

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice




############# RUN THE THINGS #####################
y_pred_im = y_pred.get_fdata()

y_true_im = y_true.get_fdata()

#y_true_im[y_true_im ==7] = 0
#y_true_im[y_true_im ==8] = 0

y_pred_mult = get_multi_class_labels(y_pred_im, len(labels), labels=labels)

y_true_mult = get_multi_class_labels(y_true_im, len(labels), labels=labels)

dice = dice_coefficient(y_true_mult, y_pred_mult)

print(K.eval(dice))