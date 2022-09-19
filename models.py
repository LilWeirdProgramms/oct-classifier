import keras.regularizers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Dense, Conv3D, MaxPooling3D, Flatten, Dropout, BatchNormalization, Activation, AlphaDropout, AveragePooling3D, Conv1D
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow import keras
import tensorflow as tf
from sklearn.decomposition import PCA

def classiRaw3D(input_size, normalizer: Normalization = None, reconstruction=False):
    #dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)
    
    init = "lecun_normal"
    binit = "lecun_normal"
    
    #input
    inp = Input(shape=input_size, dtype="float32")

    pinp = tf.keras.layers.Reshape((input_size[0] * input_size[1], input_size[2],
                             input_size[3]))(inp)  # inp= (a, b, c, channel), output (a*b, c, channel)
    pinp = tf.keras.layers.SpatialDropout2D(0.5, data_format="channels_last")(pinp)
    pinp = tf.keras.layers.Reshape((input_size[0], input_size[1], input_size[2], input_size[3]))(pinp)
    pinp = tf.keras.layers.GaussianNoise(0.3)(pinp)

    conv = pinp
    conv = Conv3D(32, 3, strides=1, activation="selu", padding="same", kernel_initializer=init, bias_initializer=binit
                 , kernel_regularizer=keras.regularizers.l2(l2=0.01)
                  )\
        (conv)
    conv = AlphaDropout(0.2)(conv)
    conv = MaxPooling3D((4, 2, 2))(conv)
    conv = Conv3D(64, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l2()
                  )(conv)
    conv = AlphaDropout(0.2)(conv)
    # conv = Conv3D(64, 2, strides=2 ,activation="selu", padding="same", kernel_initializer=init, bias_initializer=binit
    #              # , kernel_regularizer=keras.regularizers.l1()
    #               )(conv)
    conv = MaxPooling3D((4, 2, 2))(conv)
    # conv = Conv3D(32, 3, strides=2, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit
    #              , kernel_regularizer=keras.regularizers.l2()
    #               )(conv)
    conv = Conv3D(128, 3, activation="selu", padding="same", kernel_initializer=init, bias_initializer=binit
                 , kernel_regularizer=keras.regularizers.l2()
                  )(conv)
    conv = AlphaDropout(0.2)(conv)
    conv = MaxPooling3D((4, 2, 2))(conv)
    conv = Conv3D(256, 3, strides=1, activation="selu", padding="same", kernel_initializer=init, bias_initializer=binit
                 , kernel_regularizer=keras.regularizers.l2()
                  )\
        (conv)
    #conv = AlphaDropout(0.2)(conv)
    #conv = AveragePooling3D(2)(conv)
    #conv = Dropout(0.2)(conv)
    conv = Flatten()(conv)
    #dense = Dropout(0.2)(flat)
    # conv = AlphaDropout(0.2)(conv)
    # denseO = Dense(128, activation="selu", use_bias=True, kernel_initializer=init, bias_initializer=binit
    #                , kernel_regularizer=keras.regularizers.l2()
    #                )(conv)
    #denseO = AlphaDropout(0.3)(conv)
    denseO = Dense(1, activation="linear", use_bias=True
                   , kernel_regularizer=keras.regularizers.l2()
                   )(conv)

    outp = denseO

    #model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"]
                  #,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
    )
    model.summary()

    return model


def classiRaw3Dmnist(input_size, normalizer: Normalization = None, reconstruction=False):
    # dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)

    init = "lecun_normal"
    binit = "lecun_normal"

    # input
    inp = Input(shape=input_size, dtype="float32")
    conv = inp

    # pinp = tf.keras.layers.Reshape((input_size[0] * input_size[1], input_size[2],
    #                                 input_size[3]))(inp)  # inp= (a, b, c, channel), output (a*b, c, channel)
    # pinp = tf.keras.layers.SpatialDropout2D(0.1, data_format="channels_last")(pinp)
    # pinp = tf.keras.layers.Reshape((input_size[0], input_size[1], input_size[2], input_size[3]))(pinp)
    # pinp = tf.keras.layers.GaussianNoise(0.1)(conv)
    # conv = pinp

    conv = Conv3D(64, 3, strides=1, activation="selu", padding="same", kernel_initializer=init, bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l2()
                  )(conv)
    conv = AlphaDropout(0.2)(conv)
    conv = MaxPooling3D((2, 2, 2))(conv)
    conv = Conv3D(128, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l2()
                  )(conv)
    conv = AlphaDropout(0.2)(conv)
    conv = MaxPooling3D((2, 2, 2))(conv)
    conv = Conv3D(256, 3, activation="selu", padding="same", kernel_initializer=init, bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l2()
                  )(conv)
    conv = AlphaDropout(0.2)(conv)
    conv = MaxPooling3D((2, 2, 2))(conv)
    conv = Conv3D(256, 3, activation="selu", padding="same", kernel_initializer=init, bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l2()
                  )(conv)
    conv = Flatten()(conv)
    denseO = Dense(1, activation="linear", use_bias=True
                   , kernel_regularizer=keras.regularizers.l2()
                   )(conv)
    outp = denseO

    # model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"]
                  # ,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
    )
    model.summary()

    return model

# def mil_metric(y_true, y_pred):
#     prediction = tf.reshape(y_pred, (-1, 3))
#     pooled_predicition = tf.reduce_max(prediction, axis=1)
#     label = tf.reshape(y_true, (-1, 3))
#     pooled_label = tf.reduce_mean(label, axis=1)
#     #m = tf.keras.metrics.AUC()
#     #m.update_state(pooled_predicition, pooled_label)
#     return m.result()

class MilMetric(tf.keras.metrics.AUC):
    def __init__(self, name=None, **kwargs):
        super(MilMetric, self).__init__(name="mil_metric", from_logits=True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        prediction = tf.reshape(y_pred, (-1, 2))
        pooled_predicition = tf.reduce_max(prediction, axis=1)
        label = tf.reshape(y_true, (-1, 2))
        pooled_label = tf.reduce_mean(label, axis=1)
        super().update_state(pooled_label, pooled_predicition, sample_weight)

def classiRaw3Dmnist_small(input_size, normalizer: Normalization = None, reconstruction=False):
    # dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)

    init = tf.keras.initializers.RandomNormal(stddev=0.1)
    binit = tf.keras.initializers.RandomNormal(stddev=0.1)
    init = tf.keras.initializers.HeNormal()
    binit = tf.keras.initializers.HeNormal()

    # input
    inp = Input(shape=input_size, dtype="float32")
    conv = inp

    # pinp = tf.keras.layers.Reshape((input_size[0] * input_size[1], input_size[2],
    #                                 input_size[3]))(inp)  # inp= (a, b, c, channel), output (a*b, c, channel)
    # pinp = tf.keras.layers.SpatialDropout2D(0.1, data_format="channels_last")(pinp)
    # pinp = tf.keras.layers.Reshape((input_size[0], input_size[1], input_size[2], input_size[3]))(pinp)
    #pinp = tf.keras.layers.GaussianNoise(0.1)(conv)
    #conv = pinp
    conv = keras.layers.GaussianNoise(0.02)(conv)
    l2_value = 0.01
    conv = AlphaDropout(0.05)(conv)
    conv = Conv3D(16, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l2(l2=l2_value)
                  )(conv)
    conv = Conv3D(16, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l2(l2=l2_value)
                  )(conv)
    conv = MaxPooling3D((2, 2, 2))(conv)
    conv = AlphaDropout(0.05)(conv)
    conv = Conv3D(32, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                 , kernel_regularizer=keras.regularizers.l1_l2(l2=l2_value, l1=1e-6)
                  )(conv)
    # conv = Conv3D(32, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
    #               bias_initializer=binit
    #              , kernel_regularizer=keras.regularizers.l2(l2=l2_value)
    #               )(conv)
    # conv = Dropout(0.2)(conv)
    conv = MaxPooling3D((2, 2, 2))(conv)
    conv = Conv3D(64, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l1_l2(l2=l2_value, l1=0)
                  )(conv)
    # conv = MaxPooling3D((2, 2, 2))(conv)
    # conv = Conv3D(64, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
    #               bias_initializer=binit
    #               , kernel_regularizer=keras.regularizers.l1_l2(l2=l2_value, l1=0)
    #               )(conv)
    # conv = Conv3D(128, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
    #               bias_initializer=binit
    #               , kernel_regularizer=keras.regularizers.l1_l2(l2=l2_value, l1=0)
    #               )(conv)
    # conv = Conv3D(256, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
    #               bias_initializer=binit
    #               , kernel_regularizer=keras.regularizers.l1_l2(l2=l2_value, l1=0)
    #               )(conv)
    # conv = Conv3D(64, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
    #               bias_initializer=binit
    #               , kernel_regularizer=keras.regularizers.l2(l2=l2_value)
    #               )(conv)
    #conv = MaxPooling3D((2, 2, 2))(conv)

    # conv = Dropout(0.2)(conv)
    # conv = MaxPooling3D((2, 2, 2))(conv)
    # conv = Conv3D(512, 3, strides=1, activation="relu", padding="same", kernel_initializer=init,
    #               bias_initializer=binit
    #               , kernel_regularizer=keras.regularizers.l2()
    #               )(conv)
    #conv = keras.layers.GlobalAveragePooling3D()(conv)
    conv = Flatten()(conv)
    conv = AlphaDropout(0.3)(conv)
    # conv = Dense(16, activation="relu"
    #                , kernel_regularizer=keras.regularizers.l2()
    #                )(conv)
    denseO = Dense(1, activation="linear"
                   , kernel_regularizer=keras.regularizers.l2(l2=l2_value)
                   )(conv)
    outp = denseO

    # model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(learning_rate=6e-5), loss=keras.losses.BinaryCrossentropy(from_logits=True,
                                                                                           label_smoothing=0.01
                                                                                           ),
                  metrics=["accuracy", MilMetric()]
                  # ,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
                  )
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss = tf.keras.losses.MeanAbsoluteError(),
    #               metrics=["accuracy"]
    #               # ,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
    #               )
    model.summary()

    return model

def classiRaw3Dmnist_1dconv(input_size, normalizer: Normalization = None, reconstruction=False):
    # dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)

    init = tf.keras.initializers.HeNormal()
    binit = tf.keras.initializers.HeNormal()
    # TODO: Don't use glorot normal; maybe also not random Normal

    # input
    inp = Input(shape=input_size, dtype="float32")
    conv = inp

    # pinp = tf.keras.layers.Reshape((input_size[0] * input_size[1], input_size[2],
    #                                 input_size[3]))(inp)  # inp= (a, b, c, channel), output (a*b, c, channel)
    # pinp = tf.keras.layers.SpatialDropout2D(0.1, data_format="channels_last")(pinp)
    # pinp = tf.keras.layers.Reshape((input_size[0], input_size[1], input_size[2], input_size[3]))(pinp)
    #pinp = tf.keras.layers.GaussianNoise(0.1)(conv)
    #conv = pinp

    # conv = tf.keras.layers.Reshape((input_size[0] * input_size[1] * input_size[2], input_size[3]))(conv)
    conv = Permute((2, 3, 1, 4))(conv)
    conv = Conv1D(16, 16, strides=16, activation="selu", padding="valid", kernel_initializer=init, use_bias=False
              , kernel_regularizer=keras.regularizers.l2()
              )(conv)
    conv = Permute((1, 2, 4, 3))(conv)
    # conv = tf.keras.layers.Reshape((input_size[0], input_size[1], input_size[2], input_size[3]))(conv)
    # conv = tf.keras.layers.Reshape((16, 16, 16, 1))(conv)
    conv = Dropout(0.2)(conv)
    # conv = MaxPooling3D((2, 2, 2))(conv)
    conv = Conv3D(64, 3, strides=3, activation="selu", padding="same", kernel_initializer=init
                  , kernel_regularizer=keras.regularizers.l2()
                  )(conv)
    # conv = AlphaDropout(0.2)(conv)
    conv = Dropout(0.2)(conv)
    # conv = MaxPooling3D((2, 2, 2))(conv)
    conv = Conv3D(128, 3, strides=3, activation="selu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                 , kernel_regularizer=keras.regularizers.l2()
                  )(conv)
    # conv = Dropout(0.2)(conv)
    # conv = MaxPooling3D((2, 2, 2))(conv)
    # conv = Conv3D(256, 3, strides=1, activation="selu", padding="same", kernel_initializer=init,
    #               bias_initializer=binit
    #               , kernel_regularizer=keras.regularizers.l2()
    #               )(conv)
    # conv = Dropout(0.2)(conv)
    # conv = MaxPooling3D((2, 2, 2))(conv)
    # conv = Conv3D(512, 3, strides=1, activation="relu", padding="same", kernel_initializer=init,
    #               bias_initializer=binit
    #               , kernel_regularizer=keras.regularizers.l2()
    #               )(conv)
    conv = Flatten()(conv)
    denseO = Dense(1, activation="linear"
                   , kernel_regularizer=keras.regularizers.l2()
                   )(conv)
    outp = denseO

    # model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"]
                  # ,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
                  )
    model.summary()

    return model

def classiRaw3Dold2(input_size, normalizer: Normalization = None, reconstruction=False):
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

    if reconstruction:
        size_doutp = np.int32(np.floor(input_size[0] / 2))
        dense1 = Permute((4, 2, 3, 1))(
            conv_inp)  # dense layer connects input densely along last dimension; reorder dimesions; (None, 102, 102, 1536) * (1536, 736)
        dense1 = Dense(size_doutp, activation="relu", use_bias=False, kernel_initializer=init, bias_initializer=binit)(
            dense1)
        conv_inp = Permute((4, 2, 3, 1))(dense1)

    conv = conv_inp
    conv = Conv3D(16, 3, activation="relu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l1(l1=0.001)
                  )(conv)
    conv = BatchNormalization()(conv)
    conv = Conv3D(16, 3, activation="relu", padding="same", kernel_initializer=init,
                  bias_initializer=binit
                  , kernel_regularizer=keras.regularizers.l1(l1=0.001)
                  )(conv)
    conv = BatchNormalization()(conv)
    conv = Conv3D(16, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit) \
        (conv)
    conv = MaxPooling3D(pool_size=(2, 2, 2))(conv)
    # create nconv downsampling layers
    conv = BatchNormalization()(conv)
    conv = Conv3D(32, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit) \
        (conv)
    conv = BatchNormalization()(conv)
    conv = Conv3D(32, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit) \
        (conv)
    conv = MaxPooling3D(pool_size=(2, 2, 2))(conv)
    # conv = Dropout(0.2)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv3D(64, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit) \
        (conv)
    conv = BatchNormalization()(conv)
    conv = Conv3D(64, 3, activation="relu", padding="same", kernel_initializer=init, bias_initializer=binit) \
        (conv)
    # conv = MaxPooling3D(pool_size=(4, 4, 4))(conv)
    # flatten and fully connected layer
    flat = Flatten()(conv)
    denseO = Dense(1, activation="linear", use_bias=True, kernel_initializer=init, bias_initializer=binit,
                   kernel_regularizer=keras.regularizers.l1(l1=0.001))(flat)

    # output
    outp = denseO

    # model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(lr=1e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True)
                  # ,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
                  )
    model.summary()

    return model

def classiRaw3Dold(input_size, normalizer: Normalization = None, reconstruction=False):
    # dataformat: samples/A-scan x fast-axis x slow-axis x channels (unused)

    init = "glorot_normal"
    binit = "glorot_normal"

    # input
    inp = Input(shape=input_size, dtype="float32")
    conv_inp = inp
    if normalizer:
        conv_inp = normalizer(conv_inp)

    # settings
    m = 1
    nconv = 3

    if reconstruction:
        size_doutp = np.int32(np.floor(input_size[0] / 2))
        dense1 = Permute((4, 2, 3, 1))(
            conv_inp)  # dense layer connects input densely along last dimension; reorder dimesions
        dense1 = Dense(size_doutp, activation="relu", use_bias=True, kernel_initializer=init,
                       bias_initializer=binit
                       )(
            dense1)
        conv_inp = Permute((4, 2, 3, 1))(dense1)

    conv = conv_inp
    # create nconv downsampling layers
    for i in range(1, nconv + 1):
        #conv = BatchNormalization()
        conv = Conv3D(16 * m * i, 3, activation="relu", padding="same", kernel_initializer=init, use_bias=True,
                      bias_initializer=binit
                      ) \
            (conv)
        conv = MaxPooling3D(pool_size=(2, 2, 2))(conv)

    # flatten and fully connected layer
    flat = Flatten()(conv)
    denseO = Dense(1, activation="linear", use_bias=True, kernel_initializer=init,
                   bias_initializer=binit
                   #,kernel_regularizer=keras.regularizers.l1(l1=0.01)
                   )(flat)

    # output
    outp = denseO

    # model
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=Adam(lr=0.0001), loss=keras.losses.BinaryCrossentropy(from_logits=True)
                  # ,metrics=[keras.metrics.SparseCategoricalCrossentropy()]
                  )
    model.summary()

    return model


def test_model(inp):
    ## input layer
    input_layer = Input(inp)
    ## convolutional layers
    x = Conv3D(filters=8, kernel_size=(3, 3, 3), use_bias=False, padding='Same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), use_bias=False, padding='Same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ## Pooling layer
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)  # the pool_size (2, 2, 2) halves the size of its input

    ## convolutional layers
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), use_bias=False, padding='Same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), use_bias=False, padding='Same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ## Pooling layer
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.25)(x)  # No more BatchNorm after this layer because we introduce Dropout

    x = Flatten()(x)

    ## Dense layers
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(units=1, activation='linear')(x)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer, name="3D-CNN")
    model_name = model.name

    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    # "Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models which combines the best properties of the AdaGrad and RMSProp algorithms.
    # It provides an optimization algorithm that can handle sparse gradients on noisy problems. The default configuration parameters do well on most problems.""
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=Adam(1e-4),  # default: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
                  metrics=['acc'])

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
