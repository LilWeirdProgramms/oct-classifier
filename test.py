import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logging.basicConfig(filename='model_training.log', encoding='utf-8', level=logging.DEBUG)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss=loss,
        metrics=["sparse_categorical_accuracy"],
    )
    return model

with tf.device('/cpu:0'):
    loss = keras.losses.SparseCategoricalCrossentropy()
    model = get_compiled_model()

    inputs = dict(digits=x_train)
    outputs = dict(predictions=y_train)

    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(200)

    # Only use the 100 batches per epoch (that's 64 * 100 samples)
    model.fit(train_dataset, epochs=5)