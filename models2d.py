import tensorflow as tf
import tensorflow.keras as k


def my_resnet_model(input_shape, trainable=False):
    inp, out = input_layer(input_shape)

    resnet_model = k.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    resnet_model.trainable = trainable
    out = resnet_model(out)

    out = k.layers.Flatten()(out)
    out = k.layers.Dropout(0.2)(out)
    out = k.layers.Dense(1, kernel_regularizer=k.regularizers.l2())(out)
    model = compile_model(inp, out)
    return model


# TODO: TRY 50/50 classes
def image2d_full(input_shape):
    init = tf.keras.initializers.RandomNormal()
    binit = tf.keras.initializers.RandomNormal()

    inp, out = input_layer(input_shape)
    for i in range(1, 6):
        out = k.layers.Conv2D(32*i, 3, strides=2, activation="selu", padding="same",
                              kernel_initializer=init, bias_initializer=binit
                              #, kernel_regularizer=k.regularizers.l2()
            )(out)
        #out = k.layers.BatchNormalization()(out)
        #if not i % 2:
    out = k.layers.GlobalAvgPool2D()(out)
    out = k.layers.Dropout(0.5)(out)
    out = k.layers.Dense(2, activation="linear")(out)
    model = compile_model(inp, out)
    return model


def image2d_mil(input_shape):
    init = "lecun_normal"
    binit = "lecun_normal"

    inp, out = input_layer(input_shape)
    for i in range(1, 4):
        out = k.layers.Conv2D(64*i, 3, strides=1, activation="selu", padding="same",
                              kernel_initializer=init, bias_initializer=binit,
                              kernel_regularizer=k.regularizers.l2(l2=0.001))\
            (out)
        out = k.layers.AlphaDropout(0.2)(out)
        out = k.layers.MaxPooling2D(2)(out)

    out = k.layers.Flatten()(out)
    out = k.layers.Dense(1, kernel_regularizer=k.regularizers.l2())(out)
    model = compile_model(inp, out)
    return model


def input_layer(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    #inp = k.layers.GaussianNoise(0.1)(inp)
    #pre = k.layers.AlphaDropout(0.1)(inp)
    return inp, inp


def compile_model(input, output):
    model = tf.keras.Model(input, output)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none"),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    with tf.device("cpu:0"):
        my_resnet_model((2044, 2048, 3), trainable=False)
        image2d_full((2044, 2048, 1))
    #image2d_mil((102, 102, 1))
