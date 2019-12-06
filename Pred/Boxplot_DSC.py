#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:14:22 2019

@author: uqmtottr
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle 

# Load data from pickle files
# Validation data - UMC, MAG, MPRAGE and TSE
with open('/scratch/cai/DEEPSEACAT/data/20191107_leaky_ReLu/VAL_Performance_Metrics.pkl', "rb") as openfile: 
    val_dice = pickle.load(openfile)
    val_dice_mean_std = pickle.load(openfile)
    val_dice_label = pickle.load(openfile)
    val_dice_label_mean_std = pickle.load(openfile)
# Test data - UMC, MAG, MPRAGE and TSE
with open('/scratch/cai/DEEPSEACAT/data/20191107_leaky_ReLu/Performance_Metrics.pkl', "rb") as openfile: 
    test_dice = pickle.load(openfile)
    test_dice_mean_std = pickle.load(openfile)
    test_dice_label = pickle.load(openfile)
    test_dice_label_mean_std = pickle.load(openfile)

# MAG data - both MPRAGE and TSE
with open('/scratch/cai/DEEPSEACAT/data/20191118_leaky_ReLu_MAG/Performance_Metrics.pkl', "rb") as openfile: 
    mag_dice = pickle.load(openfile)
    mag_dice_mean_std = pickle.load(openfile)
    mag_dice_label = pickle.load(openfile)
    mag_dice_label_mean_std = pickle.load(openfile)
# UMC data - both MPRAGE and TSE
with open('/scratch/cai/DEEPSEACAT/data/20191118_leaky_ReLu_UMC/Performance_Metrics.pkl', "rb") as openfile: 
    umc_dice = pickle.load(openfile)
    umc_dice_mean_std = pickle.load(openfile)
    umc_dice_label = pickle.load(openfile)
    umc_dice_label_mean_std = pickle.load(openfile)
# MPRAGE data - both MAG and UMC
with open('/scratch/cai/DEEPSEACAT/data/20191118_leaky_ReLu_MPRAGE/Performance_Metrics.pkl', "rb") as openfile: 
    mp_dice = pickle.load(openfile)
    mp_dice_mean_std = pickle.load(openfile)
    mp_dice_label = pickle.load(openfile)
    mp_dice_label_mean_std = pickle.load(openfile)
# TSE data - both MAG and UMC
with open('/scratch/cai/DEEPSEACAT/data/20191118_leaky_ReLu_TSE/Performance_Metrics.pkl', "rb") as openfile: 
    tse_dice = pickle.load(openfile)
    tse_dice_mean_std = pickle.load(openfile)
    tse_dice_label = pickle.load(openfile)
    tse_dice_label_mean_std = pickle.load(openfile)

#Arrays of all label values of all subjects
val = np.reshape(val_dice_label,[20,7])
test = np.reshape(test_dice_label,[11,7])

mag = np.reshape(mag_dice_label,[5,7])
mag_notail = np.delete(mag, np.s_[6:7], axis=1)
umc = np.reshape(umc_dice_label,[6,7])
umc_notail = np.delete(umc, np.s_[6:7], axis=1)

mprage = np.reshape(mp_dice_label,[11,7])
tse = np.reshape(tse_dice_label,[11,7])

### BOXPLOT ###
'''
# MPRAGE vs. TSE
dice_label_array = val
dice_label1_array = test


box_plot_data=[dice_label_array[:,0], dice_label_array[:,1], dice_label_array[:,2], dice_label_array[:,3], dice_label_array[:,4], dice_label_array[:,5], dice_label_array[:,6], np.mean(dice_label_array,axis=1)] 
box_plot_data1=[dice_label1_array[:,0], dice_label1_array[:,1], dice_label1_array[:,2], dice_label1_array[:,3], dice_label1_array[:,4], dice_label1_array[:,5], dice_label1_array[:,6], np.mean(dice_label1_array,axis=1)] 
box=plt.boxplot(box_plot_data, labels=['ERC', 'Sub','CA1','CA2', 'DG', 'CA3', 'Tail', 'Mean'], patch_artist=True, showfliers=False, positions=np.array(range(len(box_plot_data)))*2.0-0.4, widths = 0.5) 
box1=plt.boxplot(box_plot_data1, labels=['', '','','', '', '', '', ''], patch_artist=True, showfliers=False, positions=np.array(range(len(box_plot_data1)))*2.0+0.2, widths = 0.5)
plt.title('Dice Similarity Coefficient (DSC) for Validation and Test')
plt.ylabel('DSC')
colors1 = ['#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c'] #["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
colors = ['#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8'] #["#25daa4", "#fd8c35", "#9e9bca", "#f28cc1", "#85d728", "#fdd568", "#c38b22", "#999999"] 
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
for patch, color in zip(box1['boxes'], colors1):
    patch.set_facecolor(color)
for median in box1['medians']:
    median.set(color='black')
plt.legend([box["boxes"][0], box1["boxes"][0]], ['Validation set', 'Test set'], loc='lower left')
plt.show()


# UMC vs. MAG
dice_label_array = umc_notail
dice_label1_array = mag_notail


box_plot_data=[dice_label_array[:,0], dice_label_array[:,1], dice_label_array[:,2], dice_label_array[:,3], dice_label_array[:,4], dice_label_array[:,5], np.mean(dice_label_array,axis=1)] #, dice_label_array[:,6]
box_plot_data1=[dice_label1_array[:,0], dice_label1_array[:,1], dice_label1_array[:,2], dice_label1_array[:,3], dice_label1_array[:,4], dice_label1_array[:,5], np.mean(dice_label1_array,axis=1)] #, dice_label1_array[:,6]
box=plt.boxplot(box_plot_data, labels=['ERC', 'Sub','CA1','CA2', 'DG', 'CA3', 'Mean'], patch_artist=True, showfliers=False, positions=np.array(range(len(box_plot_data)))*2.0-0.4, widths = 0.5) # , 'Tail'
box1=plt.boxplot(box_plot_data1, labels=['', '','','', '', '', ''], patch_artist=True, showfliers=False, positions=np.array(range(len(box_plot_data1)))*2.0+0.2, widths = 0.5) # , ''
plt.title('Dice Similarity Coefficient (DSC) for UMC and MAG')
plt.ylabel('DSC')
colors1 = ['#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c'] #["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
colors = ['#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8'] #["#25daa4", "#fd8c35", "#9e9bca", "#f28cc1", "#85d728", "#fdd568", "#c38b22", "#999999"] 
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
for patch, color in zip(box1['boxes'], colors1):
    patch.set_facecolor(color)
for median in box1['medians']:
    median.set(color='black')
plt.legend([box["boxes"][0], box1["boxes"][0]], ['UMC', 'MAG'], loc='lower right')
plt.show()
'''

# MPRAGE vs. TSE
dice_label_array = mprage
dice_label1_array = tse


box_plot_data=[dice_label_array[:,0], dice_label_array[:,1], dice_label_array[:,2], dice_label_array[:,3], dice_label_array[:,4], dice_label_array[:,5], dice_label_array[:,6], np.mean(dice_label_array,axis=1)] 
box_plot_data1=[dice_label1_array[:,0], dice_label1_array[:,1], dice_label1_array[:,2], dice_label1_array[:,3], dice_label1_array[:,4], dice_label1_array[:,5], dice_label1_array[:,6], np.mean(dice_label1_array,axis=1)] 
box=plt.boxplot(box_plot_data, labels=['ERC', 'Sub','CA1','CA2', 'DG', 'CA3', 'Tail', 'Mean'], patch_artist=True, showfliers=False, positions=np.array(range(len(box_plot_data)))*2.0-0.4, widths = 0.5) 
box1=plt.boxplot(box_plot_data1, labels=['', '','','', '', '', '', ''], patch_artist=True, showfliers=False, positions=np.array(range(len(box_plot_data1)))*2.0+0.2, widths = 0.5)
plt.title('Dice Similarity Coefficient (DSC) for MP-RAGE and TSE')
plt.ylabel('DSC')
colors1 = ['#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c', '#e41a1c'] #["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
colors = ['#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8', '#377eb8'] #["#25daa4", "#fd8c35", "#9e9bca", "#f28cc1", "#85d728", "#fdd568", "#c38b22", "#999999"] 
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
for patch, color in zip(box1['boxes'], colors1):
    patch.set_facecolor(color)
for median in box1['medians']:
    median.set(color='black')
plt.legend([box["boxes"][0], box1["boxes"][0]], ['MP-RAGE', 'TSE'], loc='lower right')
plt.show()

