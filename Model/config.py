#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:29:31 2019

@author: uqdlund
"""

import os

src_path = '/scratch/cai/tom_shaw/test_12000_20200810'
model_path = os.path.join(src_path, 'test_12000_20200810')
if not os.path.exists(src_path):
    os.mkdir(src_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

# Build config dictionary that might be needed later
config = dict()
config["depth"] = 5
config["strided_conv_size"] = (2, 2, 2)     # Size for the strided convolutional operations
config["image_shape"] = (176, 144, 128)     # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = [32,32,32]                # None = Train on the whole image, switch to specific dimensions if patch extraction is needed
config["labels"] = (0, 1, 2, 3, 4, 5, 6, 8)       # the label numbers on the input image, should the 0 label be included??
config["n_labels"] = len(config["labels"])  # Amount of labels
config["weights"] = [0.01,3,2,2,20,3,12,4]  # subfield weights, must match number of labels or be None, note training and val Dice, won't be between 0-1 in training, but can be calculated via highest val dice from the training is equal to Short_Dice output of mean dice from validation cases
config["all_modalities"] = ["tse", "mprage"]     # Declare all available modalities
config["training_modalities"] = config["all_modalities"]    # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])  # Configures number of channels via number of modalities
if "patch_shape" in config and config["patch_shape"] is not None:       # Determine input shape, based on patch or not
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["dilation_block"]    = True
config["n_dil_block"]       = 1             # Must be at least 1 lower than depth
config["residual"]          = True
config["dense"]             = True
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True          # if False, will use upsampling instead of deconvolution
config["n_base_filters"] = 64   # Tested at 32, no OOM
config["batch_size"] = 16       # Tested at 32
config["validation_batch_size"] = 12
config["n_epochs"] = 200                # cutoff the training after this many epochs
config["patience"] = 5                  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 25               # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 1e-5
config["learning_rate_drop"] = 0.5                                  # factor by which the learning rate will be reduced
config["learning_rate_epochs"] = None   # Number of epochs after which the learning rate will drop.
config["validation_split"] = 0.9                                    # portion of the data that will be used for training
#config["flip"] = False                                              # augments the data by randomly flipping an axis during
#config["permute"] = False               # data shape must be a cube. Augments the data by permuting in various directions
#config["distort"] = None                                            # switch to None if you want no distortion
#config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0                             # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (12, 12, 12)                # randomly offset the first patch index by up to this offset
config["skip_blank"] = True                                         # if True, then patches without any target will be skipped

config['logging_file'] = os.path.join(src_path, 'training.log')
config['data_path'] = '/scratch/cai/tom_shaw/data/data_config_flipped_full/'
config["data_file"] =       os.path.join(src_path, 'train_val.hdf5')    # Typically hdf5 file
config["model_file"] =      os.path.join(src_path, 'model.h5')          # If you have a model it will load model, if not it will save as this name
config["training_file"] =   os.path.join(src_path, 'training_ids.pkl')  # Same
config["validation_file"] = os.path.join(src_path, 'validation_ids.pkl')
config["overwrite"] = False  # If True, will overwrite previous files. If False, will use previously written files.