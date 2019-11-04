#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:31:11 2019

@author: uqmtottr
"""

# Count number of voxels of each label in all images and calculate the average


from glob import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


data_path = glob('/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/data_for_network/seg/*.nii.gz')


seg = []

l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []

res = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(data_path)):
    seg.append(nib.load(data_path[i]))
    get_seg = seg[i].get_fdata()
    arr = np.array(get_seg)
    x = arr.copy()
    
    res[0] += len(x[x == 0])
    res[1] += len(x[x == 1])
    res[2] += len(x[x == 2])
    res[3] += len(x[x == 3])
    res[4] += len(x[x == 4])
    res[5] += len(x[x == 5])
    res[6] += len(x[x == 6])
    res[7] += len(x[x == 7])
    res[8] += len(x[x == 8])
    
    l0.append(len(x[x==0]))
    l1.append(len(x[x==1]))
    l2.append(len(x[x==2]))
    l3.append(len(x[x==3]))
    l4.append(len(x[x==4]))
    l5.append(len(x[x==5]))
    l6.append(len(x[x==6]))
    l7.append(len(x[x==7]))
    l8.append(len(x[x==8]))
    
avg_l0 = res[0]/200
avg_l1 = res[1]/200
avg_l2 = res[2]/200
avg_l3 = res[3]/200
avg_l4 = res[4]/200
avg_l5 = res[5]/200
avg_l6 = res[6]/200
avg_l7 = res[7]/200
avg_l8 = res[8]/200

#Calculate volume instead of number of voxel

l0_volume = []
for num in l0:
    l0_volume.append(num*0.35**3)
    
l1_volume = []
for num in l1:
    l1_volume.append(num*0.35**3)
    
l2_volume = []
for num in l2:
    l2_volume.append(num*0.35**3)
    
l3_volume = []
for num in l3:
    l3_volume.append(num*0.35**3)
    
l4_volume = []
for num in l4:
    l4_volume.append(num*0.35**3)
    
l5_volume = []
for num in l5:
    l5_volume.append(num*0.35**3)
    
l6_volume = []
for num in l6:
    l6_volume.append(num*0.35**3)
    
l7_volume = []
for num in l7:
    l7_volume.append(num*0.35**3)
    
l8_volume = []
for num in l8:
    l8_volume.append(num*0.35**3)

#Create box plots over class (labels) distribution 
#Remember that label 7 is no longer included due to the fact that cysts (labels 7) are in both hippocampus and amygdala in MAG dataset and we are only interested in hippocampus

#all labels with background
box_plot_data=[l0_volume, l1_volume, l2_volume, l3_volume, l4_volume, l5_volume, l6_volume, l8_volume]
box=plt.boxplot(box_plot_data, labels=['Background', 'ERC','Sub','CA1','CA2', 'DG', 'CA3', 'Tail'], patch_artist=True)
plt.title('Distribution of Labels')
plt.ylabel('Volume [$mm^3$]')
colors = ["#666666", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()

#all labels, no background, with fliers (outliers) - for article
box_plot_data=[l1_volume, l2_volume, l3_volume, l4_volume, l5_volume, l6_volume, l8_volume]
box=plt.boxplot(box_plot_data, labels=['ERC','Sub','CA1','CA2', 'DG', 'CA3', 'Tail'], patch_artist=True, showfliers=False)
plt.title('Distribution of Labels')
plt.ylabel('Volume [$mm^3$]')
plt.yticks(np.arange(0, 2250, step=250))
colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()

#all labels, no background, without fliers (outliers) - for abstract
box_plot_data=[l1_volume, l2_volume, l3_volume, l4_volume, l5_volume, l6_volume, l8_volume]
box=plt.boxplot(box_plot_data, labels=['ERC','Sub','CA1','CA2', 'DG', 'CA3', 'Tail'], patch_artist=True, showfliers=False)
plt.title('Distribution of Labels')
plt.ylabel('Volume [$mm^3$]')
plt.yticks(np.arange(0, 2250, step=250))
colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


box_plot_data=[l0_volume]
box=plt.boxplot(box_plot_data, labels=['Background'], patch_artist=True)
plt.ylabel('Volume [$mm^3$]')
plt.title('Distribution of Background')
colors = ["#666666"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


box_plot_data=[l1_volume]
box=plt.boxplot(box_plot_data, labels=['ERC'], patch_artist=True)
plt.ylabel('Volume [$mm^3$]')
plt.title('Distribution of ERC')
colors = ["#1b9e77"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


box_plot_data=[l2_volume]
box=plt.boxplot(box_plot_data, labels=['Sub'], patch_artist=True)
plt.ylabel('Volume [$mm^3$]')
plt.title('Distribution of Sub')
colors = ["#d95f02"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


box_plot_data=[l3_volume]
box=plt.boxplot(box_plot_data, labels=['CA1'], patch_artist=True)
plt.title('Distribution of CA1')
plt.ylabel('Volume [$mm^3$]')
colors = ["#7570b3"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


box_plot_data=[l4_volume]
box=plt.boxplot(box_plot_data, labels=['CA2'], patch_artist=True)
plt.title('Distribution of CA2')
plt.ylabel('Volume [$mm^3$]')
colors = ["#e7298a"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


box_plot_data=[l5_volume]
box=plt.boxplot(box_plot_data, labels=['DG'], patch_artist=True)
plt.title('Distribution of DG')
plt.ylabel('Volume [$mm^3$]')
colors = ["#66a61e"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


box_plot_data=[l6_volume]
box=plt.boxplot(box_plot_data, labels=['CA3'], patch_artist=True)
plt.title('Distribution of CA3')
plt.ylabel('Volume [$mm^3$]')
colors = ["#e6ab02"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


box_plot_data=[l8_volume]
box=plt.boxplot(box_plot_data, labels=['Tail'], patch_artist=True)
plt.title('Distribution of Tail')
plt.ylabel('Volume [$mm^3$]')
colors = ["#a6761d"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
plt.show()


