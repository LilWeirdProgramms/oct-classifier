import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Dense, Conv3D, MaxPooling3D, Flatten, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf

def classiRaw3D(input_size, normalizer: Normalization = None, reconstruction=True):
    #dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)
    
    init = "glorot_normal"
    binit = "glorot_normal"
    
    #input
    inp = Input(shape=input_size, dtype="float32")
    conv_inp = inp
    if normalizer:
        conv_inp = normalizer(conv_inp)

    #settings
    m = 2

    if reconstruction:
        size_doutp = np.int32(np.floor(input_size[0]/2))
        dense1 = Permute((4, 2, 3, 1))(conv_inp) #dense layer connects input densely along last dimension; reorder dimesions
        dense1 = Dense(size_doutp, activation="relu", use_bias=False, kernel_initializer=init, bias_initializer=binit)(dense1)
        conv_inp = Permute((4, 2, 3, 1))(dense1)

    conv = conv_inp
    conv = Conv3D(16, 3, activation="relu", padding="same", kernel_initializer=init,
                  bias_initializer=binit, kernel_regularizer=keras.regularizers.l1(l1=0.01))(conv)
    conv = Conv3D(16, 3, activation="relu", padding="same", kernel_initializer=init,
                  bias_initializer=binit, kernel_regularizer=keras.regularizers.l1(l1=0.01))(conv)
    conv = MaxPooling3D(pool_size=(2, 2, 2))(conv)
    #create nconv downsampling layers
    conv = Conv3D(32, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit)\
        (conv)
    conv = Conv3D(32, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit)\
        (conv)
    conv = MaxPooling3D(pool_size=(2, 2, 2))(conv)
    conv = Dropout(0.2)(conv)
    conv = Conv3D(64, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit)\
        (conv)
    conv = Conv3D(64, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit)\
        (conv)
    conv = MaxPooling3D(pool_size=(4, 4, 4))(conv)
    #flatten and fully connected layer
    flat = Flatten()(conv)
    denseO = Dense(1, activation="linear", use_bias=True, kernel_initializer=init, bias_initializer=binit,
                   kernel_regularizer=keras.regularizers.l1(l1=0.001))(flat)
    
    #output
    outp = denseO

    #model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(lr=1e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True)
                  #,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
    )
    model.summary()

    return model


def classiRaw3Dold(input_size, normalizer: Normalization = None, reconstruction=True):
    # dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)

    init = "glorot_normal"
    binit = "glorot_normal"

    # input
    inp = Input(shape=input_size, dtype="float32")
    conv_inp = inp
    if normalizer:
        conv_inp = normalizer(conv_inp)

    # settings
    m = 2
    nconv = 7

    if reconstruction:
        size_doutp = np.int32(np.floor(input_size[0] / 2))
        dense1 = Permute((4, 2, 3, 1))(
            conv_inp)  # dense layer connects input densely along last dimension; reorder dimesions
        dense1 = Dense(size_doutp, activation="relu", use_bias=False, kernel_initializer=init, bias_initializer=binit)(
            dense1)
        conv_inp = Permute((4, 2, 3, 1))(dense1)

    conv = conv_inp
    # create nconv downsampling layers
    for i in range(1, nconv + 1):
        conv = Conv3D(16 * m * int(i ** 0.5), 3, activation="relu", padding="same", kernel_initializer=init,
                      bias_initializer=binit) \
            (conv)
        if not i % 2:
            conv = MaxPooling3D(pool_size=(2, 2, 2))(conv)

    conv = MaxPooling3D(pool_size=(3, 3, 3))(conv)
    # flatten and fully connected layer
    flat = Flatten()(conv)
    denseO = Dense(1, activation="linear", use_bias=True, kernel_initializer=init, bias_initializer=binit,
                   kernel_regularizer=keras.regularizers.l1(l1=0.01))(flat)

    # output
    outp = denseO

    # model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(lr=1e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True)
                  # ,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
                  )
    model.summary()

    return model


# Klasse um zusatz Infos hineinzugeben?


def multi_gpu_raw_3D(input_size, reconstruction):
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        multi_model = classiRaw3D(input_size, reconstruction=reconstruction)

    return multi_model


class RawClassifier:

    def __init__(self):
        self.gpu = len(tf.config.list_physical_devices('GPU'))

    def model(self):
        if self.gpu:
            return classiRaw3D
        else:
            return classiRaw3D


def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def model2():
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    value_input = tf.keras.Input(shape=(None,), dtype='int32')

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(value_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(lr=1e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True)
                  #,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
    )
    model.summary()

    return model
