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
import tensorflow as tf
from keras import backend as K
import nibabel as nib
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"



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

## Directional Boundary Dice
## Edges found in the ground truth (y_true), Find 3x3x3 cube around these voxels
## and compute Dice for comparison with the voxels in the SAME SPACE in the segmented result (y_pred)
def Directional_Boundary_Dice(y_true, y_pred):
#    y_true = y_true.get_fdata()
#    y_pred = y_pred.get_fdata()
    
    # Find edges
    imax=(sp.ndimage.maximum_filter(y_true,size=3)!=y_true)
    imin=(sp.ndimage.minimum_filter(y_true,size=3)!=y_true)
    icomb=np.logical_or(imax,imin)
    
    y_true_edges = np.where(icomb,y_true,0)
    y_true_edges_idx = np.argwhere(y_true_edges)
    n_edge_points = len(y_true_edges_idx)
    true_subset = []
    pred_subset = []
    dice_results = []
    
    for dA in range(len(y_true_edges_idx)):
        for i in range(-1,2):
            true_temp_subset = y_true[
                         y_true_edges_idx[dA][0]-1:y_true_edges_idx[dA][0]+2,
                         y_true_edges_idx[dA][1]-1:y_true_edges_idx[dA][1]+2,
                         y_true_edges_idx[dA][2]+i
                         ]
            pred_temp_subset = y_pred[
                         y_true_edges_idx[dA][0]-1:y_true_edges_idx[dA][0]+2,
                         y_true_edges_idx[dA][1]-1:y_true_edges_idx[dA][1]+2,
                         y_true_edges_idx[dA][2]+i
                         ]
            true_subset.append(true_temp_subset)
            pred_subset.append(pred_temp_subset)
        true_subset      =   np.asarray(true_subset)
        pred_subset   =   np.asarray(pred_subset)
    
        dice_results.append(dice_coefficient(pred_subset, true_subset))
        
        true_subset = []
        pred_subset = []
    DBD_sum = K.sum(K.flatten(dice_results))
    DBD = DBD_sum/n_edge_points
    return DBD, DBD_sum, n_edge_points
    
# Symmetric Boundary Dice
# Two way Directional Boundary Dice

def Symmetric_Boundary_Dice(y_true, y_pred):
    true_DBD, true_sum_DBD, true_n_edge_points = Directional_Boundary_Dice(y_true, y_pred)
    pred_DBD, pred_sum_DBD, pred_n_edge_points = Directional_Boundary_Dice(y_pred, y_true)
    
    SBD = (true_sum_DBD+pred_sum_DBD)/(true_n_edge_points+pred_n_edge_points)
    return SBD

def SBD_loss(y_true, y_pred):
    return -Symmetric_Boundary_Dice(y_true,y_pred)
'''


path = '/afm01/Q1/Q1219/data/ashs_atlas_umcutrecht/train/train000/tse_native_chunk_left_seg.nii.gz' 

seg = nib.load(path)

seg_im_data = seg.get_fdata()
seg = seg_im_data

## Test purposes remove other labels except for 1 and background
thresh_idx = seg >1
seg_1label = seg.copy()
seg_1label[thresh_idx] = 0
seg_1label_2 = seg_1label.copy()
seg_1label_2[:,:,0:60] = 0

SBD = Symmetric_Boundary_Dice(seg_1label_2, seg_1label)

f = open("Metrics_output.txt", "a")
f.write(SBD)
f.close()
'''
