# -*- coding: utf-8 -*-
"""
Based on: https://github.com/ellisdg/3DUnetCNN/

Author: Daniel Ramsing Lund
mail: dlund13@student.aau.dk - Danielramsing@gmail.com

"""

import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Dropout
from keras.optimizers import Adam

from Metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge 
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    
    :param pool_size: Pool size for the max pooling operations.
    
    :param n_labels: Number of binary labels that the model is learning.
    
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling (Now strided convolutions)
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            # Create pooling layer --> replaced with strided convolution strides = (2,2,2)
            current_layer = create_convolution_block(input_layer=layer2, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, strides=(2,2,2)) #                               MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    # Note, backwards iteration to hit correct layers to concatenate
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        # Implement Dilated_fusion_block here #
        #               Code                  # 1 or 2 blocks? # Make it mutable? 
        # Concatenate [layer_depth][1] since [2] is strided convolution/max-pooling
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)        
        
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)
    
def create_dilated_fusion_block(input_layer, n_filters, dilation_depth, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    # Create dilated blocks depending on depth
    for dil_rate in range(dilation_depth):
        if dil_rate == 0:
            layer = dilated_couple(n_filters, kernel, padding=padding, strides=strides, dilation_rate=2**dil_rate, dropout=0.5)(input_layer)
        else:
            layer = dilated_couple(n_filters, kernel, padding=padding, strides=strides, dilation_rate=2**dil_rate, dropout=0.5)(layer)


# NOTE! Check if Layer_Depth is available in this function or declare as input variable
# Consider SpatialDropout3D instead of Dropout, if performance is bad, especially good in early layers
def dilated_couple(input_layer, n_filters, dilation_rate, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, dropout = 0.0):
    # First conv has kernel (1,1,1) to reduce feature maps and thereby computational load
    layer = dilated_conv(n_filters, kernel = (1,1,1), padding=padding, strides=strides, dilation_rate=dilation_rate)(input_layer)
    layer = dilated_conv(n_filters/(2**layer_depth), kernel, padding=padding, strides=strides, dilation_rate=dilation_rate)(layer)
    
    layer = Dropout(dropout)(layer)
    concat = concatenate([layer, input_layer], axis=1)
    
    layer = dilated_conv(n_filters, kernel = (1,1,1), padding=padding, strides=strides, dilation_rate=dilation_rate)(concat)
    layer = dilated_conv(n_filters/(2**layer_depth), kernel, padding=padding, strides=strides, dilation_rate=dilation_rate)(layer)
    
    layer = Dropout(dropout)(layer)
    
    final_layer = concatenate([layer, concat], axis=1)
    return final_layer

# Single dilated conv w. relu activation
def dilated_conv(input_layer, n_filters, dilation_rate, batch_normalization=False, kernel=(3,3,3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, dropout = False):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, dilation_rate=dilation_rate)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)