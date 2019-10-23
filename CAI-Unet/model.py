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
from keras.utils import multi_gpu_model

from Metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge 
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape, strided_conv_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False, 
                  dilation_block = False, n_dil_block=1, residual=False, dense=False, depth=4, n_base_filters=32, 
                  include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the (pool size)/(strided_conv_size) to the power of the depth of the UNet, that is strided_conv_size^depth.
    
    :param strided_conv_size: Size for the strided convolution operations
    
    :param n_labels: Number of binary labels that the model is learning.
    
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    
    :param deconvolution: Boolean - If set to True, will use transpose convolution (deconvolution) instead of up-sampling. This
    increases the amount memory required during training. (default is False)
    
    :param dilation_block: Boolean - if set to True, will construct dilated block between encoder-decoder as in https://doi.org/10.3389/fninf.2019.00030 
    depends on 'n_dil_block' for amount of dilated blocks to implement (default is False)
    
    :param n_dil_blocks: Number of dilated blocks to implement - starts from lowest possible layer i.e. depth-1 (default is 1 block)
    
    :param residual: Boolean - if set to True, will employ residual connections on each depth-level, currently matching n_filters for strided convolutions in encoder
    and 1x1x1 convolution to normalize input in decoder (default is False)
    
    :param dense: Boolean - if set to True, will employ dense connections
    
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more (pooling layers)/(strided convolutional layers)
    will be added to the model. Lowering the depth will reduce the amount of memory required for training. (default is 4 (which is also where most testing occurred))
    
    :param n_base_filters: Number of filters that the first layer in the convolution network will have. Following layers will contain a 
    multiple of this number (doubling for each depth-level). Lowering this number will reduce the amount of memory required to train the model. (default is 32)
    
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric. (default is False)
    
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    
    :param batch_normalization: Boolean - If set to true incoorporates batch normalization in every convolutional layer (default is False)
    
    :param activation_name: relevant activation function in final layer before inference/prediction
    
    
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling (Now strided convolutions)
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        
        ## Residual and Dense implementation ##
        
        if residual and dense:
                        
            concat = concatenate([current_layer, layer1], axis=1)
            
            layer2 = create_convolution_block(input_layer=concat, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, act_man = True)
            
            concat = concatenate([current_layer, layer1, layer2], axis=1)
            
            #OBS: Det har giver fejlen 'Operands could not be broadcast together with shapes (130, 176, 144, 128) (2, 176, 144, 128)'
            #Har proevet at fixe med nedenstaaende (og sa selvfolgelig add med norm_conv i stedet for concat), men det virker ikke rigtig
            
            # Needs to be either concat._keras_shape[1] and input layer = current_layer
            # OR current_layer.keras_shape[1] and input_layer = concat 
            # This determines whether we get lots of feature maps for the next step or few (130 vs 2) for the first level, should not be a problem in later levels.
            norm_conv = create_convolution_block(n_filters=concat._keras_shape[1],input_layer=current_layer, batch_normalization=batch_normalization, kernel = (1,1,1))
            
            layer2 = add([norm_conv, concat])
            layer2 = Activation('relu')(layer2)
        
        ## Residual implementation ##
        elif residual and layer_depth>0:
            
            layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, act_man = True)
            
            layer2 = add([layer2, current_layer])
            layer2 = Activation('relu')(layer2)
        
        ## Dense implementation ##
        elif dense:
            
            concat = concatenate([current_layer, layer1], axis=1)
            
            layer2 = create_convolution_block(input_layer=concat, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
            layer2 = concatenate([current_layer, layer1, layer2], axis=1)

        else:
            layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
            
        # for all levels except lowest, create strided convolutional layer   
        if layer_depth < depth - 1:
            # Create pooling layer --> replaced with strided convolution strides = (2,2,2)##      REMOVE +1 here depending on choice for residual ##
            current_layer = create_convolution_block(input_layer=layer2, n_filters=n_base_filters*(2**(layer_depth+1)),
                                          batch_normalization=batch_normalization, strides=strided_conv_size) #MaxPooling3D(pool_size=pool_size)(layer2)
            
            # Keep track of layers in each level for later upsampling
            levels.append([layer1, layer2, current_layer])
        # For last layer, do ##NOT## include strided convolution and just have the two normal convolutions
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution (Transposed convolution) or up-sampling using 'levels' list
    # Note, backwards iteration necessary to hit correct layers, in the correct order, for concatenation
    for layer_depth in range(depth-2, -1, -1):
        # NOTE that n_filters is determined by the original n_filter levels, NOT the ones that are affected by dense and residual connections
        up_convolution = create_up_convolution(current_layer, strided_conv_size=strided_conv_size, deconvolution=deconvolution,
                                            n_filters=levels[layer_depth][0]._keras_shape[1])
        
        ## Dilated_fusion_block ##
        # ##Legacy comment## modify for-loop here if wanted on multiple levels, should be permutable, just specify
        # >= depth - (depth-1) for all except top layer, and depth > depth-depth
        if layer_depth >= (depth-(n_dil_block+1)) and dilation_block:
            dil_out = create_dilated_fusion_block(input_layer=levels[layer_depth][1], n_filters=levels[layer_depth][0]._keras_shape[1], layer_depth=layer_depth, dilation_depth=3)
            concat = concatenate([up_convolution, dil_out], axis=1)
        
        else:
            # SKIP-CONNECTION # Concatenate [layer_depth][1] since [2] is strided convolution/max-pooling
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)        
        
        
        #OBS: residual and dense skal staa foerst, naar det nedadgaaende virker
        ## Residual and dense implementation ##
        if residual and dense:
            current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, 
                                                 batch_normalization=batch_normalization)
            # Dense 1
            concat_dense = concatenate([current_layer, concat], axis=1)
            current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization, act_man=True)
            # Dense 2            
            current_layer = concatenate([current_layer, concat, concat_dense], axis=1)
            
            # 1x1x1 layer to normalize number of feature maps
            norm_conv = create_convolution_block(n_filters=concat._keras_shape[1],#levels[layer_depth][1]
                                                 input_layer=current_layer, 
                                                 batch_normalization=batch_normalization, 
                                                 kernel = (1,1,1))
            
            current_layer = add([norm_conv, concat])
            current_layer = Activation('relu')(current_layer)
        
        
        ## Residual implementation  ##
        elif residual:
            # 1x1x1 layer to normalize number of feature maps
            norm_conv = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, 
                                                 batch_normalization=batch_normalization, 
                                                 kernel = (1,1,1))
            current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, 
                                                 batch_normalization=batch_normalization)
            current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization, act_man=True)
            current_layer = add([current_layer, norm_conv])
            current_layer = Activation('relu')(current_layer)
            
        ## Dense implementation ##    
        elif dense:
            current_layer = create_convolution_block(n_filters=levels[layer_depth][0]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
            # Dense 1
            concat_dense = concatenate([current_layer, concat], axis=1)
            current_layer = create_convolution_block(n_filters=levels[layer_depth][0]._keras_shape[1],
                                                 input_layer=concat_dense,
                                                 batch_normalization=batch_normalization)
            # Dense 2
            current_layer = concatenate([current_layer, concat, concat_dense], axis=1)
        
        
        
        else:
            current_layer = create_convolution_block(n_filters=levels[layer_depth][0]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
            current_layer = create_convolution_block(n_filters=levels[layer_depth][0]._keras_shape[1],
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
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=[dice_coefficient]) #loss=dice_coefficient_loss OR 'binary_crossentropy' OR 'categorical_crossentropy', metrics=['accuracy']
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, act_man = False):
    """
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :param strides:
    :param instance_normalization: Different type of regularization, that can be employed if not wanting batch_normalization
    :param act_man: Boolean - if residual implementation is wanted, addition must happen before activation, therefore this allows for manual activation
    
    :return: A Keras convolution block. Can be normalized (not default) and activated (default)
    """
    if strides!=(1,1,1):
        layer = Conv3D(n_filters, kernel, padding=padding, kernel_initializer='he_normal', strides=strides, name='Strided_conv'+str(input_layer._keras_shape[1]))(input_layer)
    elif kernel!=(3,3,3):
        layer = Conv3D(n_filters, kernel, padding=padding, kernel_initializer='he_normal', strides=strides, name='1x1x1_'+str(input_layer._keras_shape[1]))(input_layer)
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
    Helper function - Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param strided_conv_size: the strided_conv_size parameter used in the strided convolution operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(strided_conv_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def create_up_convolution(input_layer, n_filters, strided_conv_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    '''
    Construct upsampling block.
    :param deconvolution: Boolean - If True use deconvolution (transposed3D convolution), if False use Upsampling (default is False)
    '''
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides, kernel_initializer='he_normal', name='DeConv'+str(n_filters))(input_layer)
    else:
        return UpSampling3D(size=strided_conv_size)(input_layer)
    
def create_dilated_fusion_block(input_layer, n_filters, layer_depth=2, dilation_depth=3, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    '''
    :param layer_depth: Number denoting which depth-level to construct dilated fusion block in, used to name layers in later function (should always be specified, but default is 2)
    :param dilation_depth: Number determining length of dilated_fusion_block (default is 3). Increases dilation rate (in increments of 2**('dilation_depth')) for each additional dilation depth
    '''
    # Create dilated blocks depending on depth. dil_rate goes: 1 --> 2 --> 4 --> 8 etc.
    for dil_rate in range(dilation_depth):
        if dil_rate == 0:
            layer = dilated_couple(input_layer, n_filters, layer_depth, kernel=kernel, padding=padding, strides=strides, dilation_rate=2**dil_rate, dropout=0.5)
        else:
            layer = dilated_couple(layer, n_filters, layer_depth, kernel=kernel, padding=padding, strides=strides, dilation_rate=2**dil_rate, dropout=0.5)            
    return layer



# Consider SpatialDropout3D instead of Dropout, if performance is bad, especially good in early layers
# Give input layer and outputs concatenated layer of two sets of two dil convs.
# NOTE - Can be made neater with for loop
# NOTE - There is a LOT of Dropout here. Additionally a bug has been reported where dropout was not properly applied. Check here if regularization is bonkers
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