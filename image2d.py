import InputList
import tensorflow as tf
import logging
from DatasetBaseClass import DatasetBaseClass
import numpy as np


class ImageDataset(DatasetBaseClass):

    def __init__(self, image_type="angio", learning_type="supervised", augmentation=True, pseduocolor=False, image=True,
                 val_split=0.2):
        """

        :param image_type: Must be one of ["angio", "struct", "combined", "overlay"]
        :param learning_type: Must be one of ["supervised", "mil"]
        """
        super().__init__(val_split)
        self.is_image = image
        self.image_type = image_type
        self.learning_type = learning_type
        self.augmentation = augmentation
        self.dataset_augmentation = None
        self.pseudocolor = pseduocolor

    def create_dataset(self, input_files, deterministic=True, augmented=False):
        #self.dataset_augmentation = augmented
        paths = [path_label[0] for path_label in input_files]
        labels = [path_label[1] for path_label in input_files]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if not deterministic:
            dataset = dataset.shuffle(len(paths))
        dataset = dataset.map(self._parse_image, num_parallel_calls=8).prefetch(4)
        return dataset

    def get_training_files(self):
        converted_input_files = list(map(self._create_image_path_from_raw, InputList.training_files))
        converted_input_files = super().remove_invalid_file_paths(converted_input_files, self.is_image)
        return converted_input_files

    def get_test_files(self):
        converted_input_files = list(map(self._create_image_path_from_raw, InputList.testing_files))
        return super().remove_invalid_file_paths(converted_input_files)

    def _create_image_path_from_raw(self, raw_tuple: tuple[str, int]) -> tuple:
        if self.image_type == "angio":
            front_ident = InputList.angio_ident[0]
            back_indent = InputList.angio_ident[1]
        elif self.image_type == "struct":
            front_ident = InputList.struct_ident[0]
            back_indent = InputList.struct_ident[1]
        elif self.image_type == "combined":
            front_ident = InputList.struct_ident[0]
            back_indent = InputList.struct_ident[1]
        elif self.image_type == "overlay":
            front_ident = InputList.overlay_ident[0]
            back_indent = InputList.overlay_ident[1]
        else:
            raise NotImplementedError("image_type of ImageDataset must be one of [angio, struct]")
        image_tuple = (raw_tuple[0].replace(InputList.binary_ident[0], front_ident).replace(InputList.binary_ident[1],
                                                                                            back_indent), raw_tuple[1])
        return image_tuple

    def _parse_image(self, filename, label):
        if self.learning_type == "supervised":
            image_string = tf.io.read_file(filename)
            image = tf.io.decode_png(image_string, channels=1)
        elif self.learning_type == "mil":
            image_string = tf.io.read_file(filename)
            image = tf.io.decode_png(image_string, channels=1)
            tf.image.extract_patches(images=image,
                                     sizes=[1, 102, 102, 1],
                                     strides=[1, 102, 102, 1],
                                     rates=[1, 1, 1, 1],
                                     padding='VALID')
        else:
            raise NotImplementedError("learning_type of ImageDataset must be one of [supervised, mil]")
        image = self._preprocess_image(image)
        return image, label

    def _preprocess_image_and_label(self, image, label):
        return self._preprocess_image(image), label

    def _preprocess_image(self, image):
        #image = image / 255
        if self.augmentation:
            crop_at = 300
            image = image[crop_at:-crop_at, crop_at:-crop_at]
            #image = tf.image.random_flip_left_right(image)
            #image = tf.image.random_flip_up_down(image)
            #image = tf.image.random_brightness(image, max_delta=0.2)
            #image = tf.image.random_contrast(image, 0.7, 1.2)
            #image = tf.image.random_brightness(image, max_delta=0.4)
            #image = tf.image.random_contrast(image, 0.5, 1.5)
        image = tf.image.per_image_standardization(image)
        return image


def __test_image_to_raw():
    id = ImageDataset()
    image_path = id._create_image_path_from_raw(("raw_2323.bin", 1))
    assert image_path[0] == "retina_2323.png"
    id = ImageDataset("struct")
    image_path = id._create_image_path_from_raw(("raw_2323.bin", 1))
    assert image_path[0] == "enf_2323.png"


import matplotlib.pyplot as plt


def __test_dataset():
    id = ImageDataset()
    train_ds, val_ds, _, _ = id.get_training_datasets()
    numy_plots = 12
    fig, ax = plt.subplots(numy_plots, 1, figsize=(14, 110))
    cnt = 0
    for elem in train_ds.repeat().take(12):
        ax[cnt].imshow(elem[0].numpy(), cmap="gray")
        print(elem[1])
        cnt += 1
    plt.savefig("results/preprocessed_images")


if __name__ =="__main__":
    logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.DEBUG)
    with tf.device("cpu:0"):
        __test_dataset()
    #__test_image_to_raw()
