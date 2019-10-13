#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:41:50 2019

@author: uqdlund

Test to build the 3D unet model, can maybe serve as main later on.
"""

from model import unet_model_3d
from Metrics import dice_coefficient
import os
import glob

# Build config dictionary that might be needed later
config = dict()
config["depth"] = 4
config["strided_conv_size"] = (2, 2, 2)     # Size for the strided convolutional operations
config["image_shape"] = (176, 144, 128)     # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None                # Train on the whole image, switch to specific dimensions if patch extraction is needed
config["labels"] = (1, 2, 3, 4, 5, 6)       # the label numbers on the input image
config["n_labels"] = len(config["labels"])  # Amount of labels
config["all_modalities"] = ["t1", "t2"]     # Declare all available modalities
config["training_modalities"] = config["all_modalities"]    # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])  # Configures number of channels via number of modalities
if "patch_shape" in config and config["patch_shape"] is not None:       # Determine input shape, based on patch or not
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["dilation_block"] = False
config["n_dil_block"]       = 1             # Must be at least 1 lower than depth
config["residual"]          = True
config["dense"] = True
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True          # if False, will use upsampling instead of deconvolution
config["n_base_filters"] = 64
config["batch_size"] = 6
config["validation_batch_size"] = 12
config["n_epochs"] = 100                # cutoff the training after this many epochs
config["patience"] = 5                  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 15               # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5                                  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8                                    # portion of the data that will be used for training
config["flip"] = False                                              # augments the data by randomly flipping an axis during
config["permute"] = False               # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None                                            # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0                              # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)                # randomly offset the first patch index by up to this offset
config["skip_blank"] = True                                         # if True, then patches without any target will be skipped

config['data_path'] = '../../../../afm01/Q1/Q1219/data/ashs_atlas_umcutrecht_7t_20170810/train/train-all/'
config["data_file"] = os.path.abspath("[Your datafile here ]")      # Typically hdf5 file
config["model_file"] = os.path.abspath("[Your model name here]")    # If you have a model it will load model, if not it will save as this name
config["training_file"] = os.path.abspath("training_ids.pkl")       # Same
config["validation_file"] = os.path.abspath("validation_ids.pkl")   # Same
config["overwrite"] = False  # If True, will overwrite previous files. If False, will use previously written files.

# Some code to fetch data #


# Hmmmm have a look at what this actually fetches
def fetch_training_data_files():
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(config['data_path'],'*')):#os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files



# Some code to construct generators for training, validation and test #
training_data_files = fetch_training_data_files()




model = unet_model_3d(input_shape = config["input_shape"], 
                      strided_conv_size = config["strided_conv_size"],
                      n_labels = config["n_labels"],
                      dilation_block = config["dilation_block"],
                      n_dil_block = config["n_dil_block"],
                      initial_learning_rate = config["initial_learning_rate"],
                      deconvolution = config["deconvolution"],
                      residual = config["residual"],
                      dense = config["dense"],
                      n_base_filters = config["n_base_filters"])