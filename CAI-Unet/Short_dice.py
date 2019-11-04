#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:12:50 2019

@author: uqdlund
"""

import numpy as np
from keras import backend as K
import nibabel as nib

y_pred = nib.load('/scratch/cai/DEEPSEACAT/data/20191030_p64_b16_noOverlap/prediction/validation_case_101/prediction.nii.gz')


y_true = nib.load('/scratch/cai/DEEPSEACAT/data/20191030_p64_b16_noOverlap/prediction/validation_case_101/truth.nii.gz')

labels = (1, 2, 3, 4, 5, 6, 8)

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

def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])

def volume_similarity(y_true, y_pred):
    y_true_vol = [0, 0, 0, 0, 0, 0, 0]
    y_pred_vol = [0, 0, 0, 0, 0, 0, 0]
    true_whole_vol = []
    pred_whole_vol = []
    vs = []    
  
    for num_of_labels in range(y_true_mult.shape[3]):
        y_true_vol[num_of_labels] = 0.35**3 * len(y_true_mult[:,:,:,num_of_labels][y_true_mult[:,:,:,num_of_labels] == 1])
        
        y_pred_vol[num_of_labels] = 0.35**3 * len(y_pred_mult[:,:,:,num_of_labels][y_pred_mult[:,:,:,num_of_labels] == 1])
        
        true_whole_vol[0] += 0.35**3 * len(y_true_mult[:,:,:,num_of_labels][y_true_mult[:,:,:,num_of_labels] == 1])
        pred_whole_vol[0] += 0.35**3 * len(y_pred_mult[:,:,:,num_of_labels][y_pred_mult[:,:,:,num_of_labels] == 1])
    
    vs_whole_hippo = 1 - np.absolute(np.absolute(pred_whole_vol[0]) - np.absolute(true_whole_vol[0]))/(np.absolute(pred_whole_vol[0]) + np.absolute(true_whole_vol[0]))

    for label_num in range(len(y_true_vol)):
        vs_val = 1 - np.absolute(np.absolute(y_pred_vol[label_num]) - np.absolute(y_true_vol[label_num]))/(np.absolute(y_pred_vol[label_num]) + np.absolute(y_true_vol[label_num]))
        # The calculation of vs with a prediction of 50 mm^3 and a truth of 100 mm^3 gives a vs of 66%, which makes sence according to the calculation but how does this fit to the though of 50 being 50% of 100?
        vs.append(vs_val)
    return vs, vs_whole_hippo

############# RUN THE THINGS #####################
y_pred_im = y_pred.get_fdata()

y_true_im = y_true.get_fdata()

#y_true_im[y_true_im ==7] = 0
#y_true_im[y_true_im ==8] = 0

y_pred_mult = get_multi_class_labels(y_pred_im, len(labels), labels=labels)

y_true_mult = get_multi_class_labels(y_true_im, len(labels), labels=labels)

dice = dice_coefficient(y_true_mult, y_pred_mult)

dice_label = label_wise_dice_coefficient(y_true_mult, y_pred_mult, labels)

vs = volume_similarity(y_true_mult, y_pred_mult)

print(K.eval(dice))
print(vs)