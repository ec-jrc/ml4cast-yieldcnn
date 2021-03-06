#!/usr/bin/python

"""
Model architecture definition

Multiple inputs: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
Multiple outputs: https://github.com/rahul-pande/faces-mtl/blob/master/faces_mtl_age_gender.ipynb
"""

from deeplearning.architecture_features import *
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow_addons.layers import SpatialPyramidPooling2D
import sys


# -----------------------------------------------------------------------
# ---------------------- ARCHITECTURES
# ------------------------------------------------------------------------

# -----------------------------------------------------------------------
#MM: unify architectures

def Archi_2DCNN(archi_type, Xt, Xv=None, nbunits_conv=10, kernel_size=3, strides=3, pool_size=3, pyramid_bins=[1], dropout_rate=0.,
                     nb_fc=1, nunits_fc=64, activation='sigmoid', l2_rate = 1.e-6, verbose=True):
    if not archi_type in ['MISO','SISO']:
        print('Architecture ' + str(archi_type) + ' is not defined, the program will stop.')
        sys.exit()

    # -- get the input sizes and define the input placeholder. (Input() is used to instantiate a Keras tensor)
    n_batches, image_y, image_x, n_bands = Xt.shape
    input_shape_t = (image_y, image_x, n_bands)
    Xt_input = Input(input_shape_t, name='ts_input')

    if archi_type == 'MISO':
        mv, Lv = Xv.shape
        input_shape_v = (Lv,)
        Xv_input = Input(input_shape_v, name='v_input')
    # -- parameters of the architecture
    #l2_rate = 1.e-6

    # -- nb_conv CONV layers
    Xt = Xt_input
    Xt = conv2d_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             kernel_regularizer=l2(l2_rate))
    Xt = AveragePooling2D(pool_size=pool_size, strides=strides, padding='valid')(Xt)  # does not alter n of channels
    # test version 8_MaxPool
    #Xt = MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid')(Xt)  # does not alter n of channels
    Xt = conv2d_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             kernel_regularizer=l2(l2_rate))  # keeps the same number of channels
    #Xt = GlobalAveragePooling2D(data_format='channels_last')(Xt)
    Xt = SpatialPyramidPooling2D(pyramid_bins, data_format='channels_last')(Xt)

    if archi_type == 'MISO':
        # -- Flatten
        Xt = Flatten()(Xt)
        # -- Vector inputs
        Xv = Xv_input
        Xv = Dense(nbunits_conv, activation=activation)(Xv)  # n units = n conv channels to add some balance among channels
        # -- Concatenate
        X = layers.Concatenate()([Xt, Xv])
    elif archi_type == 'SISO':
        # -- Flatten
        X = Flatten()(Xt)

    # -- Output FC layers
    for add in range(nb_fc): #can be zero (no layers)
        X = Dense(nunits_fc//pow(2, add), activation=activation)(X)
        X = Dropout(dropout_rate)(X)
    out1 = Dense(1, name='out1')(X)
    #out1 = Dense(1, activation='relu', name='out1')(X)

    # Create model
    if archi_type == 'MISO':
        model = Model(inputs=[Xt_input, Xv_input], outputs=[out1], name=f'Archi_2DCNN_MISO')
    elif archi_type == 'SISO':
        model = Model(inputs=Xt_input, outputs=[out1], name=f'Archi_2DCNN_SISO')

    if verbose:
        model.summary()
        print('Input shape', str(model.layers[0].output_shape))
        print('After conv1', str(model.layers[4].output_shape))
        print('After avgPool2D', str(model.layers[5].output_shape))
        print('After conv2', str(model.layers[9].output_shape))
        print('After PyramidPool', str(model.layers[10].output_shape))
        print('***')
    return model



def _Archi_2DCNN_MISO(Xt, Xv, nbunits_conv=10, kernel_size=3, strides=3, pool_size=3, pyramid_bins=[1], dropout_rate=0.,
                     nb_fc=1, nunits_fc=64, activation='sigmoid', verbose=True):
    # -- get the input sizes
    if isinstance(Xt, list):
        input_shape_t = (Xt[0], Xt[1])
        input_shape_v = (Xv[0],)
    else:
        n_batches, image_y, image_x, n_bands = Xt.shape
        input_shape_t = (image_y, image_x, n_bands)
        mv, Lv = Xv.shape
        input_shape_v = (Lv,)
    # -- parameters of the architecture
    l2_rate = 1.e-6

    # Define the input placeholder. (Input() is used to instantiate a Keras tensor)
    Xt_input = Input(input_shape_t, name='ts_input')
    Xv_input = Input(input_shape_v, name='v_input')

    # -- nb_conv CONV layers
    Xt = Xt_input
    Xt = conv2d_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             kernel_regularizer=l2(l2_rate))
    Xt = AveragePooling2D(pool_size=pool_size, strides=strides, padding='valid')(Xt)
    Xt = conv2d_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             kernel_regularizer=l2(l2_rate))
    #Xt = GlobalAveragePooling2D(data_format='channels_last')(Xt)
    Xt = SpatialPyramidPooling2D(pyramid_bins, data_format='channels_last')(Xt)

    # -- Flatten
    Xt = Flatten()(Xt)

    # -- Vector inputs
    Xv = Xv_input
    Xv = Dense(nbunits_conv, activation=activation)(Xv)  # n units = n conv channels to add some balance among channels

    # -- Concatenate
    X = layers.Concatenate()([Xt, Xv])

    # -- Output FC layers
    for add in range(nb_fc): #can be zero (no layers)
        X = Dense(nunits_fc//pow(2, add), activation=activation)(X)
        X = Dropout(dropout_rate)(X)

    out1 = Dense(1, activation='relu', name='out1')(X)

    # Create model.
    model = Model(inputs=[Xt_input, Xv_input], outputs=[out1], name=f'Archi_CNNw_MISO')
    if True:
        print('Input shape', str(model.layers[0].output_shape))
        print('After conv1', str(model.layers[4].output_shape))
        print('After avgPool2D', str(model.layers[5].output_shape))
        print('After conv2', str(model.layers[9].output_shape))
        print('After PyramidPool', str(model.layers[10].output_shape))
        print('***')
    if verbose:
        model.summary()
    return model


# -----------------------------------------------------------------------
def _Archi_2DCNN_SISO(Xt, nbunits_conv=10, kernel_size=3, strides=3, pool_size=3, pyramid_bins=[1], dropout_rate=0.,
                     nb_fc=1, nunits_fc=1, activation='sigmoid', verbose=True):
    # -- get the input sizes
    if isinstance(Xt, list):
        input_shape_t = (Xt[0], Xt[1])
    else:
        n_batches, image_y, image_x, n_bands = Xt.shape
        input_shape_t = (image_y, image_x, n_bands)
    # -- parameters of the architecture
    l2_rate = 1.e-6

    # Define the input placeholder.
    Xt_input = Input(input_shape_t, name='ts_input')    ##MM: Input() is used to instantiate a tensorflow.keras tensor.

    # -- nb_conv CONV layers
    Xt = Xt_input
    Xt = conv2d_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             kernel_regularizer=l2(l2_rate)) ##MM: returns nbunits_conv channels
    Xt = AveragePooling2D(pool_size=pool_size, strides=strides, padding='valid')(Xt) ##MM: does not alter n of channels
    Xt = conv2d_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             kernel_regularizer=l2(l2_rate)) #MM: keeps the same number of channels
    #Xt = GlobalAveragePooling2D(data_format='channels_last')(Xt)    #MM operate in space, so I get only one value per channel (nbunits_conv)
    Xt = SpatialPyramidPooling2D(pyramid_bins, data_format='channels_last')(Xt)

    # -- Flatten
    X = Flatten()(Xt)

    # -- Output FC layers
    for add in range(nb_fc - 1):
        X = Dense(nunits_fc//pow(2, add), activation=activation)(X)
        X = Dropout(dropout_rate)(X)

    out1 = Dense(1, activation='relu', name='out1')(X)

    # Create model.
    model = Model(inputs=Xt_input, outputs=[out1], name=f'Archi_2DCNN_SISO')
    if verbose:
        model.summary()
    return model

# EOF
