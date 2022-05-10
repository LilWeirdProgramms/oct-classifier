import matplotlib.pyplot as plt

import Callbacks
import InputList
import image2d
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import models2d
import image2d
import InputListUtils
from Postprocessor import Postprocessing
import os
import logging

def calc_class_weights(labels: np.ndarray):
    a = labels[labels == 1].size
    b = labels[labels == 0].size
    class_weights = {0: (1 / b) * (a + b) / 2, 1: (1 / a) * (a + b) / 2}
    #class_weights = {0: 1, 1: 1}
    print(f"Got {labels.size} Training Samples of which {a} are diabetic and {b} are healthy")
    print(class_weights)
    return class_weights


def image2d_run(name, pretrained_model_name=None):
    with tf.device("cpu:0"):
        image_data1 = image2d.ImageDataset(image_type="angio", learning_type="supervised")
        train_ds1, val_ds1, _, _ = image_data1.get_training_datasets()
    if pretrained_model_name is None:
        model = models2d.image2d_full((2044-600, 2048-600, 1))
    else:
        model = k.models.load_model(f'savedModels/image{pretrained_model_name}')
    train_labels = []
    for train_sample in train_ds1:
        train_labels.append(train_sample[1].numpy())
    for train_sample in train_ds1.take(4):
        plt.imshow(train_sample[0].numpy()[:, :, 0], "gray")
        plt.show()
    class_weights = calc_class_weights(np.array(train_labels))
    history = model.fit(
        train_ds1.batch(2),
        epochs=80,
        validation_data=val_ds1.batch(2),
        callbacks=Callbacks.my_image_callbacks,
        class_weight=class_weights
    )
    model.save(f'savedModels/image{name}')


def image2d_eval(name):
    #labels = np.array([i for path, i in InputList.training_files])
    #class_weights = calc_class_weights(labels)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    image_data1 = image2d.ImageDataset(image_type="angio", learning_type="supervised", augmentation=False, val_split=1)
    train_ds1, val_ds1, _, val_files = image_data1.get_training_datasets()
    model = k.models.load_model(f'savedModels/image{name}')
    model_output = model.predict(
        val_ds1.batch(1),
        verbose=1,
        use_multiprocessing=True
    )
    labels = []
    all_tings = []
    for ground_truth, predicted, file in zip(val_ds1, model_output, val_files):
        #print(f"Predicted: {predicted}, Ground Truth: {ground_truth[1]}")
        labels.append(ground_truth[1].numpy())
        if (ground_truth[1].numpy() == 1 and predicted[1] < 0) or (ground_truth[1].numpy() == 0 and predicted[0] < 0):
            #plt.imsave(f"results/p{len(labels)}.png", ground_truth[0].numpy().reshape((2044, 2048)), cmap="gray")
            print(f"\nFrom File {file}:")
            print(f"Predicted: {predicted}, Ground Truth: {ground_truth[1]}")
        all_tings.append(scce(ground_truth[1], predicted))
    #pp = Postprocessing(model_output, labels)
    #pp.binary_confusion_matrix()
    #pp.binary_confusion_matrix(name=os.path.join("/home/julius/Documents/masterarbeit/arbeit_figures/confusion_image", name))
    print(np.max(model_output))
    print(np.mean(model_output))
    print(np.mean(all_tings))
    plt.hist([elem.numpy()[0] for elem in all_tings], bins=10)
    plt.show()

import InputListUtils

def main(eval=False):
    with tf.device("cpu:0"):
        name = "8"
        file_list = []
        with open('diabetic_training_files.txt', 'r') as f:
            for item in f.read().splitlines():
                file_list.append((item, 1))
        with open('healthy_training_files.txt', 'r') as f:
            for item in f.read().splitlines():
                file_list.append((item, 0))
        file_list.extend(InputListUtils.find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location="/media/julius/My Passport/MOON1e"))
        file_list.extend(InputListUtils.find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location="/media/julius/My Passport1/MOON1e"))
        InputList.training_files = file_list[40:]
        # with tf.device('/cpu:0'):
        #     id = image2d.ImageDataset()
        #     file_list = id.get_training_files()
        #     for file in file_list:
        #         image = id._parse_image(file[0], str(file[1]))
        #         if not image[0].shape == (2044, 2048, 1):
        #             print(file[0], image[0].shape)
    if eval:
        image2d_eval(name)
    else:
        image2d_run(name)

if __name__ == "__main__":
    logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.DEBUG)
    main()
    main(eval=True)
