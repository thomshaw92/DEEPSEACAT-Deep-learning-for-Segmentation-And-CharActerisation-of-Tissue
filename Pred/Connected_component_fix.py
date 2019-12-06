#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:06:30 2019

@author: uqdlund
"""

import numpy as np

import nibabel as nib
import os
from glob import glob
import scipy.ndimage as sp

y_pred = []
y_true = []
dice = []
dice_label= []
vs = []
vs_label_wise = []



def Connected_components(src_path, labels):
    pred_cases = sorted(glob(os.path.join(src_path, '*','prediction.nii.gz')))
    for case in pred_cases:
        y_pred.append((nib.load(case)).get_fdata())
    
    for i in range(len(pred_cases)):
        #y_pred_im = y_pred[i].get_fdata() 
        y_pred_im_new = np.zeros_like(y_pred[i])
        for label in labels:
            idx = y_pred[i] == label
            labelled_mask, num_labels = sp.label(idx)
            sizes = sp.sum(idx, labelled_mask, range(num_labels + 1))
            #clipping_mask = sizes == max(sizes)
            for region in range(num_labels+1):
                if np.sum(idx[labelled_mask == region]) != max(sizes):
                    idx[labelled_mask == region] = 0
            y_pred_im_new += idx*label
        save_path = os.path.join(src_path, 'Connected_Components',pred_cases[i][pred_cases[i].index('validation_case'):len(pred_cases[i])])
        if not os.path.exists(save_path[0:len(save_path)-17]):
            os.mkdir(save_path[0:len(save_path)-17])
        nib.save(nib.Nifti1Image(y_pred_im_new, affine=None), save_path)
        #print('done with case ', i)


labels = (1,2,3,4,5,6,8)

src_path = '/scratch/cai/DEEPSEACAT/data/20191031_p32_depth5_dense/test_prediction/'
pred_cases = sorted(glob(os.path.join(src_path, '*','prediction.nii.gz')))
for case in pred_cases:
    y_pred.append(nib.load(case))
    
for i in range(len(pred_cases)):
    y_pred_im = y_pred[i].get_fdata() 
    y_pred_im_new = np.zeros_like(y_pred_im)
    for label in labels:
        idx = y_pred_im == label
        labelled_mask, num_labels = sp.label(idx)
        sizes = sp.sum(idx, labelled_mask, range(num_labels + 1))
        #clipping_mask = sizes == max(sizes)
        for region in range(num_labels+1):
            if np.sum(idx[labelled_mask == region]) != max(sizes):
                idx[labelled_mask == region] = 0
        y_pred_im_new += idx*label
    save_path = os.path.join(src_path, 'CC_new',pred_cases[i][pred_cases[i].index('validation_case'):len(pred_cases[i])])
    if not os.path.exists(save_path[0:len(save_path)-17]):
        os.mkdir(save_path[0:len(save_path)-17])
    nib.save(nib.Nifti1Image(y_pred_im_new, affine=None), save_path)
    print('done with case ', i)