#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:36:23 2019

@author: uqdlund
"""
import os
import nibabel as nib
import numpy as np
from glob import glob
import scipy as sp
import scipy.ndimage as spn 

def Connected_components(src_path, labels):
    pred_cases = sorted(glob(os.path.join(src_path, '*','prediction.nii.gz')))
    y_pred = []
    for case in pred_cases:
        y_pred.append((nib.load(case)).get_fdata())
    
    for i in range(len(pred_cases)):
        #y_pred_im = y_pred[i].get_fdata() 
        y_pred_im_new = np.zeros_like(y_pred[i])
        for label in labels:
            idx = y_pred[i] == label
            labelled_mask, num_labels = spn.label(idx)
            sizes = spn.sum(idx, labelled_mask, range(num_labels + 1))
            #clipping_mask = sizes == max(sizes)
            for region in range(num_labels+1):
                if np.sum(idx[labelled_mask == region]) != max(sizes):
                    idx[labelled_mask == region] = 0
            y_pred_im_new += idx*label
        # redefine dir_names for each case
        new_path = os.path.join(src_path, 'CC')
        save_path = os.path.join(new_path,pred_cases[i][pred_cases[i].index('validation_case'):len(pred_cases[i])])
        if not os.path.exists(save_path[0:len(save_path)-17]):
            os.mkdir(save_path[0:len(save_path)-17])
        nib.save(nib.Nifti1Image(y_pred_im_new, affine=None), save_path)
        return new_path


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

def dice_coefficient_np(y_true, y_pred, smooth=1):
    y_true_f = np.flatten(y_true)
    y_pred_f = np.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice

def label_wise_dice_coefficient(y_true, y_pred, label_index):
    lab_dice = []
    for i in range(len(label_index)):
        lab_dice.append(dice_coefficient_np(y_true[:,:,:,i], y_pred[:,:,:,i]))
    return lab_dice

def Directional_Boundary_Dice(y_true, y_pred):    
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
    
        dice_results.append(dice_coefficient_np(pred_subset, true_subset))
        
        true_subset = []
        pred_subset = []
    DBD_sum = np.sum(np.flatten(dice_results))
    DBD = DBD_sum/n_edge_points
    return DBD, DBD_sum, n_edge_points
    
# Symmetric Boundary Dice
# Two way Directional Boundary Dice
def Symmetric_Boundary_Dice(y_true, y_pred):
    true_DBD, true_sum_DBD, true_n_edge_points = Directional_Boundary_Dice(y_true, y_pred)
    pred_DBD, pred_sum_DBD, pred_n_edge_points = Directional_Boundary_Dice(y_pred, y_true)
    
    SBD = (true_sum_DBD+pred_sum_DBD)/(true_n_edge_points+pred_n_edge_points)
    return SBD

def volume_deviation(y_true, y_pred, labels):
    temp, temp2=[]
    for label in labels:
        temp.append(len(y_true[y_true==label]))
        temp2.append(len(y_pred[y_pred==label]))
    true_vol = np.asarray(temp)*0.35**3
    pred_vol = np.asarray(temp2)*0.35**3
    
    vol_dev = ((pred_vol - true_vol)/true_vol)*100
    
    return [pred_vol, true_vol, vol_dev]


def Performance_metrics(src_path, labels):
    pred_cases = sorted(glob(os.path.join(src_path, '*','prediction.nii.gz')))
    y_pred_im, y_true_im, dice,dice_label, SBD, vol = []
    for case in pred_cases:
        y_pred_im.append((nib.load(case)).get_fdata())


    true_cases = sorted(glob(os.path.join(src_path, '*','truth.nii.gz')))
    for case in true_cases:
        y_true_im.append((nib.load(case)).get_fdata())

    
    for i in range(len(pred_cases)):         
        y_pred_mult = get_multi_class_labels(y_pred_im[i], len(labels), labels=labels)
        y_true_mult = get_multi_class_labels(y_true_im[i], len(labels), labels=labels)
        
        dice.append(dice_coefficient_np(y_true_mult, y_pred_mult))
        dice_label.append(label_wise_dice_coefficient(y_true_mult, y_pred_mult, labels))
        for label in labels:
            SBD.append(y_true_mult[:,:,:,label], y_pred_mult[:,:,:,label])
            
        vol.append(volume_deviation(y_pred_im[i],y_true_im[i],labels))
        
    return dice, dice_label, SBD, vol