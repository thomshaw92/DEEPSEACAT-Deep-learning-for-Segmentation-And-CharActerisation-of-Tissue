#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:41:50 2019

@author: uqdlund

Test to build the 3D unet model, can maybe serve as main later on.
"""
import sys
import os
import glob
sys.path.append(os.getcwd()) 

from keras.utils import plot_model
from keras.models import load_model



from Model.utils import fetch_training_data_files, write_data_to_file, open_data_file, get_callbacks
from Model.generator import get_training_and_validation_generators
from Model.model import unet_model_3d


src_path = '/scratch/cai/DEEPSEACAT/data/20191107_base_Wconfig3' 
if not os.path.exists(src_path):
    os.mkdir(src_path)

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
config['data_path'] = '/scratch/cai/DEEPSEACAT/data/data_config_flipped/'
config["data_file"] =       os.path.join(src_path, 'train_val.hdf5')    # Typically hdf5 file 'train_val_flipped.hdf5' and 'train_val_100.hdf5'
config["model_file"] =      os.path.join(src_path, 'model.h5')          # If you have a model it will load model, if not it will save as this name
config["training_file"] =   os.path.join(src_path, 'training_ids.pkl')  # Same
config["validation_file"] = os.path.join(src_path, 'validation_ids.pkl')
config["overwrite"] = True  # If True, will overwrite previous files. If False, will use previously written files.



def main(overwrite=False):
    # convert input images into an hdf5 file
    # if we want to overwrite files or it doen't exist create datafile
    if overwrite or not os.path.exists(config["data_file"]):
        print('fetching data files... \n')
        training_files = fetch_training_data_files(config["data_path"], config["training_modalities"])
        print('writing data to file... \n')
        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])
        
    data_file_opened = open_data_file(config["data_file"])
    
    if not overwrite and os.path.exists(config["model_file"]):
        model = load_model(config["model_file"])
        
    else:
        # instantiate new model
        #with tf.device('/cpu:0'):
        model = unet_model_3d(input_shape = config["input_shape"], 
                  strided_conv_size = config["strided_conv_size"],
                  n_labels = config["n_labels"],
                  dilation_block = config["dilation_block"],
                  n_dil_block = config["n_dil_block"],
                  initial_learning_rate = config["initial_learning_rate"],
                  deconvolution = config["deconvolution"],
                  #include_label_wise_dice_coefficients = True,
                  residual = config["residual"],
                  dense = config["dense"],
                  n_base_filters = config["n_base_filters"])

    # print summary of model to double check, as well as save image of model
    #'''
    model.summary()
    plot_model(model, 'model_test.png', show_shapes=True)     
    #'''
    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        #permute=config["permute"],
        #augment=config["augment"],
        skip_blank=config["skip_blank"],
        weights = config["weights"] #,
        #augment_flip=config["flip"]
        #augment_distortion_factor=config["distort"]
        )

    # run training
    # train the model:
    print('fitting model...')
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=n_train_steps,
                        epochs=config['n_epochs'],
                        validation_data=validation_generator,
                        validation_steps=n_validation_steps,
                        # Callbacks go to training.log
                        callbacks=get_callbacks(config["model_file"],
                                                logging_file = config['logging_file'],
                                                initial_learning_rate=config["initial_learning_rate"],
                                                learning_rate_drop=config["learning_rate_drop"],
                                                learning_rate_epochs=config["learning_rate_epochs"],
                                                learning_rate_patience=config["patience"],
                                                early_stopping_patience=config["early_stop"]))
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
