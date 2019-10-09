#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:36:06 2019

@author: uqdlund
"""

from functools import partial
#import tensorflow as tf
import scipy as sp
import numpy as np
from keras import backend as K
import nibabel as nib

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

# Symmetric Boundary Dice implement here
'''
Test med samme segmentering to gange = dice 1, translater den ene lidt og pr'v igen = dice ~ 0.9
Find kant pixels for de 'binære' objekter og lav dice på et område omkring hver af disse, summer og divider med antallet af steder det er gjort.
repeat for den anden segmentering og ny divider    
'''

#def symmetric_boundary_dice(y_true, y_pred, neigh_radi=1):
    


#dice_coef = dice_coefficient
#dice_coef_loss = dice_coefficient_loss
path = '/afm01/Q1/Q1219/data/ashs_atlas_umcutrecht/train/train000/tse_native_chunk_left_seg.nii.gz' 

seg = nib.load(path)

seg_im_data = seg.get_fdata()
seg = seg_im_data

## Test purposes remove other labels except for 1 and background
thresh_idx = seg >1
seg_1label = seg.copy()
seg_1label[thresh_idx] = 0

#dice_results = dice_coefficient(seg_1label,seg_1label.copy())

# Find edges
imax=(sp.ndimage.maximum_filter(seg_1label,size=3)!=seg_1label)
imin=(sp.ndimage.minimum_filter(seg_1label,size=3)!=seg_1label)
icomb=np.logical_or(imax,imin)

seg_edges = np.where(icomb,seg_1label,0)

seg_edges_idx = np.argwhere(seg_edges)
seg_ground = seg.copy()

seg_subset_seg =    []
seg_subset_ground = []
dice_results = []


## Currently implemented up to what is considered Directional Boundary Dice
## Needs to be implemented for ground truth segmentation as well (essentially the other way around) 
## edges should be found in the ground truth, Find 3x3x3 cube around these voxels
## and compute Dice for comparison with the segmented result
## Currently we get a dice higher than 1 ?.?
for dA in range(len(seg_edges_idx)):
    for i in range(-1,2):
        seg_subset = seg[
                     seg_edges_idx[dA][0]-1:seg_edges_idx[dA][0]+2,
                     seg_edges_idx[dA][1]-1:seg_edges_idx[dA][1]+2,
                     seg_edges_idx[dA][2]+i
                     ]
        seg_ground_subset = seg_ground[
                     seg_edges_idx[dA][0]-1:seg_edges_idx[dA][0]+2,
                     seg_edges_idx[dA][1]-1:seg_edges_idx[dA][1]+2,
                     seg_edges_idx[dA][2]+i
                     ]
        seg_subset_seg.append(seg_subset)
        seg_subset_ground.append(seg_subset)
    seg_subset_seg      =   np.asarray(seg_subset_seg)
    seg_subset_ground   =   np.asarray(seg_subset_ground)

    dice_results.append(dice_coefficient(seg_subset_ground, seg_subset_seg))
    
    seg_subset_seg = []
    seg_subset_ground = []

overall_dice = K.eval(K.sum(dice_results))
#seg_f = K.flatten(seg)
#seg2_f = K.flatten(seg)

#intersection = K.sum(seg2_f * seg_f)