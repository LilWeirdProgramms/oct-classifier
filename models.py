import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Dense, Conv3D, MaxPooling3D, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf

def classiRaw3D(input_size, normalizer: Normalization = None, reconstruction=True):
    #dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)
    
    init = "glorot_normal"
    binit = "glorot_normal"
    
    #input
    inp = Input(shape=input_size, dtype="float32")  # TODO: sketchy way
    conv_inp = inp
    if normalizer:
        conv_inp = normalizer(conv_inp)

    #settings
    m = 2
    nconv = 4

    if reconstruction:
        size_doutp = np.int32(np.floor(input_size[0]/2))
        dense1 = Permute((4, 2, 3, 1))(conv_inp) #dense layer connects input densely along last dimension; reorder dimesions
        dense1 = Dense(size_doutp, activation="relu", use_bias=False, kernel_initializer=init, bias_initializer=binit)(dense1)
        conv_inp = Permute((4, 2, 3, 1))(dense1)

    conv = conv_inp
    #create nconv downsampling layers
    for i in range(1, nconv+1):
        conv = Conv3D(32*m*i, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit)\
            (conv)
        conv = MaxPooling3D(pool_size=(2, 2, 2))(conv)

    #flatten and fully connected layer
    flat = Flatten()(conv)
    denseO = Dense(1, activation="linear", use_bias=True, kernel_initializer=init, bias_initializer=binit,
                   kernel_regularizer=keras.regularizers.l1(l1=0.01))(flat)
    
    #output
    outp = denseO

    #model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(lr=1e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True)
                  #,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
    )
    model.summary()

    return model


# Klasse um zusatz Infos hineinzugeben?


def multi_gpu_raw_3D(input_size, normalizer, reconstruction):
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        multi_model = classiRaw3D(input_size, normalizer, reconstruction)

    return multi_model


class RawClassifier:

    def __init__(self):
        self.gpu = len(tf.config.list_physical_devices('GPU'))

    def model(self):
        if self.gpu:
            return multi_gpu_raw_3D
        else:
            return classiRaw3D
