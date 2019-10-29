#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:12:50 2019

@author: uqdlund
"""

import numpy as np
from keras import backend as K
import nibabel as nib

y_pred = nib.load('/scratch/cai/DEEPSEACAT/data/20191028_weighted_dice/prediction/validation_case_100/prediction.nii.gz')

y_true = nib.load('/scratch/cai/DEEPSEACAT/data/20191028_weighted_dice/prediction/validation_case_100/truth.nii.gz')

labels = (1, 2, 3, 4, 5, 6, 7, 8)


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)    
    
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y

def dice_coefficient(y_true, y_pred, smooth=1):
    #weights = [2,2,1,20,2,12,1,6]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice




############# RUN THE THINGS #####################
y_pred_im = y_pred.get_fdata()

y_true_im = y_true.get_fdata()

y_pred_mult = get_multi_class_labels(y_pred_im, len(labels), labels=labels)

y_true_mult = get_multi_class_labels(y_true_im, len(labels), labels=labels)

dice = dice_coefficient(y_true_mult, y_pred_mult)

print(K.eval(dice))