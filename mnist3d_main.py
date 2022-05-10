import os.path

import numpy as np
import models
import mnist3d
import Callbacks
import tensorflow as tf
import tensorflow.keras as k
from Postprocessor import Postprocessing
import matplotlib.pyplot as plt

def calc_class_weights(labels: np.ndarray):
    a = labels[labels == 1].size
    b = labels[labels == 0].size
    class_weights = {0: (1 / b) * (a + b) / 2, 1: (1 / a) * (a + b) / 2}
    #class_weights = {0: 1, 1: 1}
    print(class_weights)
    return class_weights

import utils

def image_run(batch_size, epochs, name):
    mnist_data = mnist3d.MNISTDataHandler(frequency=True)
    train_dataset, val_dataset, test_dataset = mnist_data.create_dataset()
    class_weights = calc_class_weights(mnist_data.y_train)
    model = models.classiRaw3Dmnist_1dconv((16, 16, 16, 1))
    utils.calc_receptive_field3D(model)
    history = model.fit(
        train_dataset.batch(batch_size).shuffle(mnist_data.y_train.size),
        epochs=epochs,
        validation_data=val_dataset.batch(batch_size),
        callbacks=Callbacks.my_mnist_callbacks,
        class_weight=class_weights
    )
    model.save(f'savedModels/mnist{name}')


def image_eval(name):
    mnist_data = mnist3d.MNISTDataHandler(frequency=True)
    train_dataset, val_dataset, test_dataset = mnist_data.create_dataset()
    set_to_eval = test_dataset
    model = k.models.load_model(f'savedModels/mnist{name}')
    # plt.figure()
    # plt.imshow(model.layers[2].get_weights()[0].reshape(32, 16))
    # plt.show()
    # plt.figure()
    # plt.plot(model.layers[2].get_weights()[0].flatten()[:64])
    # plt.show()
    model_output = model.predict(
        set_to_eval.batch(1),
        verbose=1,
        use_multiprocessing=True
    )
    labels = []
    for ground_truth, predicted, gt2 in zip(set_to_eval, model_output, mnist_data.y_test_org):
        print(f"Predicted: {predicted}, Ground Truth: {ground_truth[1]}, Truly: {gt2}")
        labels.append(ground_truth[1].numpy())
    pp = Postprocessing(model_output, mnist_data.y_test_org)
    pp.binary_confusion_matrix(name=os.path.join("/home/julius/Documents/masterarbeit/arbeit_figures/confusion_fourier", name))
    print(np.max(model_output))
    print(np.mean(model_output))
    print(np.mean(model_output[np.array(labels) == 1]))
    print(np.mean(model_output[np.array(labels) == 0]))


def main(eval=False):
    name = "mconv1d_i_c_nb"
    if eval:
        image_eval(name)
    else:
        image_run(2, 40, name)


if __name__ == "__main__":
    #main()
    main(eval=True)
