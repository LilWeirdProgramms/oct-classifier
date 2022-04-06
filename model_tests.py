import numpy as np
import tensorflow as tf
import tensorflow.keras as k
bunch_of_ascans = np.zeros((3, 2, 2, 1))
one_ascan = np.array([1, 2, 5])
bunch_of_ascans = bunch_of_ascans + one_ascan[:, None, None, None]
print(bunch_of_ascans[:, 1, 1])
kernel_shape = np.zeros((3, 1, 3))
kernel = np.array([3, 2, 1])
kernel = kernel_shape + kernel[:, None, None]
print(kernel.shape)
model_input = k.layers.Input((3, 2, 2, 1))
conv = k.layers.Permute((2, 3, 1, 4))(model_input)
conv = k.layers.Conv1D(3, 3, strides=3, activation="linear", padding="valid",  trainable=False, use_bias=False)(conv)
model_output = k.layers.Permute((4, 1, 2, 3))(conv)
model = k.models.Model(model_input, model_output)
model.compile(optimizer="Adam")
model.summary()
model.layers[2].set_weights([kernel])
model.predict(bunch_of_ascans.reshape((1, 3, 2, 2, 1)))
