import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Dense, Conv3D, MaxPooling3D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

def classiRaw3D(input_size=(1536, 256, 256, 1), reconstruction=True):
    #dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)
    
    init = "glorot_normal"
    binit = "glorot_normal"
    
    #input
    inp = Input(input_size, dtype="float32")
    normalized = BatchNormalization()(inp)
    #settings
    m = 1
    nconv = 5
    conv = normalized

    if reconstruction:
        size_doutp = np.int32(np.floor(input_size[0]/2))
        dense1 = Permute((4, 2, 3, 1))(normalized) #dense layer connects input densely along last dimension; reorder dimesions
        dense1 = Dense(size_doutp, activation="relu", use_bias=False, kernel_initializer=init, bias_initializer=binit)(dense1)
        dense1 = Permute((4, 2, 3, 1))(dense1)
        conv = dense1

#create nconv downsampling layers
    for i in range(1, nconv+1):
        conv = Conv3D(32*m*i, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit)(conv)
        conv = MaxPooling3D(pool_size=(2, 2, 2))(conv)
        
    #flatten and fully connected layer
    flat = Flatten()(conv)
    denseO = Dense(1, activation="softmax", use_bias=False, kernel_initializer=init, bias_initializer=binit)(flat)
    
    #output
    outp = denseO

    #model
    model = Model(input=inp, output=outp)
    model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy")
    model.summary()

    return model