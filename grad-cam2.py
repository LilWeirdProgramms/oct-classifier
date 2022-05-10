import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import os
import matplotlib.pyplot as plt


import image2d

model = k.models.load_model(f'savedModels/image10')
image_data1 = image2d.ImageDataset(image_type="angio", learning_type="supervised"
                                   , augmentation=False
                                   , val_split=1
                                   )
train_ds1, val_ds1, train_files, val_files = image_data1.get_training_datasets()
i = 0
for image, file in zip(val_ds1.batch(1), val_files):
    i += 1
    x = image[0].numpy()[:, 300:-300, 300:-300, :]
    plt.imsave(f"results/grad_cam/original{i}.png", x.reshape((1444, 1448)), cmap="gray")
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_22')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, tf.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = k.backend.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    print(f"\nFrom File {file}:")
    print("Predicted:", model_out[0], "True: ", image[1])
    plt.imsave(f"results/grad_cam/grad_cam{i}.png", np.abs(heatmap.numpy().reshape(
        last_conv_layer.shape[1:3].as_list()
    )))
