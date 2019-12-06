#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:41:50 2019

@author: uqdlund

Test to build the 3D unet model, can maybe serve as main later on.
"""
import sys
import os
sys.path.append(os.getcwd()) 

from keras.utils import plot_model
from keras.models import load_model

from Model.utils import fetch_training_data_files, write_data_to_file, open_data_file, get_callbacks
from Model.generator import get_training_and_validation_generators
from Model.model import unet_model_3d
from Model.config import config

########################### Make changes to the model in Model.config #########################################

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
