# -*- coding: utf-8 -*-
"""
Based on: https://github.com/ellisdg/3DUnetCNN/

Author: Daniel Ramsing Lund
mail: dlund13@student.aau.dk - Danielramsing@gmail.com

"""

import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Dropout, add
from keras.optimizers import Adam

from Metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge 
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape, strided_conv_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False, 
                  dilation_block = False, n_dil_block=1, residual=False, depth=4, n_base_filters=32, 
                  include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more strided convolutional
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is strided_conv_size^depth.
    
    :param strided_conv_size: Size for the strided convolution operations
    
    :param n_labels: Number of binary labels that the model is learning.
    
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    
    :param deconvolution: If set to True, will use transpose convolution (deconvolution) instead of up-sampling. This
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
        if residual and layer_depth>0:
            layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, act_man = True)
            layer2 = add([layer2, current_layer])
            layer2 = Activation('relu')(layer2)
        else:
            layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
            
        if layer_depth < depth - 1:
            # Create pooling layer --> replaced with strided convolution strides = (2,2,2)
            current_layer = create_convolution_block(input_layer=layer2, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, strides=strided_conv_size) #MaxPooling3D(pool_size=pool_size)(layer2)
            
            levels.append([layer1, layer2, current_layer])
        else:
            if residual:
                layer2 = add([layer2, current_layer])
            else:
                current_layer = layer2
            
            levels.append([layer1, layer2])

    # add levels with up-convolution (Transposed convolution) or up-sampling
    # Note, backwards iteration to hit correct layers to concatenate
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(strided_conv_size=strided_conv_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        # Dilated_fusion_block here # 
        # Currently only implemented at lowest level, modify for-loop here if wanted on multiple levels, should be permutable, just specify
        # >= depth - (depth-1) for all except top layer, and depth > depth-depth
        if layer_depth >= (depth-(n_dil_block+1)) and dilation_block:
            dil_out = create_dilated_fusion_block(input_layer=levels[layer_depth][1], n_filters=levels[layer_depth][1]._keras_shape[1], layer_depth=layer_depth, dilation_depth=3)
            concat = concatenate([up_convolution, dil_out], axis=1)
        else:
            # Concatenate [layer_depth][1] since [2] is strided convolution/max-pooling
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)        
        
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
        if residual:
            current_layer = add([current_layer, concat])

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

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss='binary_crossentropy', metrics=['accuracy']) #loss=dice_coefficient_loss, metrics=metrics
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, act_man = False):
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
    if strides!=(1,1,1):
        layer = Conv3D(n_filters, kernel, padding=padding, kernel_initializer='he_normal', strides=strides, name='Strided_conv'+str(input_layer._keras_shape[1]))(input_layer)
    else:
        layer = Conv3D(n_filters, kernel, padding=padding, kernel_initializer='he_normal', strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None and act_man is False:
        return Activation('relu')(layer)
    elif act_man is True:
        return layer
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, strided_conv_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param strided_conv_size: the strided_conv_size parameter used in the strided convolution operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(strided_conv_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, strided_conv_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides, kernel_initializer='he_normal', name='DeConv'+str(n_filters))
    else:
        return UpSampling3D(size=strided_conv_size)
    
def create_dilated_fusion_block(input_layer, n_filters, layer_depth=2, dilation_depth=3, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    # Create dilated blocks depending on depth
    for dil_rate in range(dilation_depth):
        if dil_rate == 0:
            layer = dilated_couple(input_layer, n_filters, layer_depth, kernel=kernel, padding=padding, strides=strides, dilation_rate=2**dil_rate, dropout=0.5)
        else:
            layer = dilated_couple(layer, n_filters, layer_depth, kernel=kernel, padding=padding, strides=strides, dilation_rate=2**dil_rate, dropout=0.5)            
    return layer



# Consider SpatialDropout3D instead of Dropout, if performance is bad, especially good in early layers
# Give input layer and outputs concatenated layer of two sets of two dil convs.
# NOTE - Can be mdke neater with for loop
def dilated_couple(input_layer, n_filters, layer_depth, dilation_rate=1, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, dropout = 0.0):
    # First conv has kernel (1,1,1) to reduce feature maps and thereby computational load
    layer = dilated_conv(input_layer, n_filters, kernel = (1,1,1), padding=padding, strides=strides, dilation_rate=dilation_rate, name = str(layer_depth)+'dil_rate_'+str(dilation_rate)+'_'+str(n_filters)+'_1')
    # filters should be the same as base layer here
    layer = dilated_conv(layer, n_filters//(2**2), kernel=kernel, padding=padding, strides=strides, dilation_rate=dilation_rate, name = str(layer_depth)+'dil_rate_'+str(dilation_rate)+'_'+str(n_filters//(2**2))+'_1')
    
    layer = Dropout(dropout)(layer)
    # First set of completed dilated convs
    concat = concatenate([layer, input_layer], axis=1)
    
    layer = dilated_conv(concat, n_filters, kernel = (1,1,1), padding=padding, strides=strides, dilation_rate=dilation_rate, name = str(layer_depth)+'dil_rate_'+str(dilation_rate)+'_'+str(n_filters)+'_2')
    layer = dilated_conv(layer, n_filters//(2**2), kernel=kernel, padding=padding, strides=strides, dilation_rate=dilation_rate, name = str(layer_depth)+'dil_rate_'+str(dilation_rate)+'_'+str(n_filters//(2**2))+'_2')
    
    layer = Dropout(dropout)(layer)
    # Final layer is concatenated output of first 2 dil convs and second set of 2 dil convs
    final_layer = concatenate([layer, concat], axis=1)
    return final_layer

# Single dilated conv w. relu activation
   # Give input layer and outputs layer convoluted with dilated conv
def dilated_conv(input_layer, n_filters, name='dilated_conv', dilation_rate=1, batch_normalization=False, kernel=(3,3,3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, dropout = False):
    layer = Conv3D(n_filters, kernel, kernel_initializer='he_normal', padding=padding, strides=strides, dilation_rate=dilation_rate, name=name)(input_layer)
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