import tensorflow as tf
import Callbacks
import image2d
import InputListUtils
import logging
import random
import numpy as np


class ImageDataset():
    def __init__(self, data_list=None, validation_split=True):
        if data_list is None:
            self.data_list = ImageDataset.get_file_list()
        else:
            self.data_list = data_list
        self.validation_split = validation_split
        self.dataset_train: tf.data.Dataset = None
        self.dataset_val: tf.data.Dataset = None
        self.create_dataset_from_file_list()

    @staticmethod
    def get_file_list():
        healthy_path_list = InputListUtils.get_file_list_from_folder("data/healthy_images", 0)
        diabetic_path_list = InputListUtils.get_file_list_from_folder("data/diabetic_images", 1)
        logging.info(f"Healthy Files: {len(healthy_path_list)}")
        logging.info(f"Diabetic Files: {len(diabetic_path_list)}")
        combined_list = []
        combined_list.extend(healthy_path_list)
        combined_list.extend(diabetic_path_list)
        random.shuffle(combined_list)
        return combined_list

    def create_dataset_from_file_list(self):
        x_train = []
        y_train = []
        id = image2d.ImageDataset()
        for path, label in self.data_list:
            image, _ = id._parse_image(path, label)
            x_train.append(image)
            y_train.append(label)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if self.validation_split:
            self.dataset_train = tf.data.Dataset.from_tensor_slices((x_train[:-30], y_train[:-30]))
            self.dataset_val = tf.data.Dataset.from_tensor_slices((x_train[-30:], y_train[-30:]))
        else:
            self.dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    @staticmethod
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=1.2)
        image = tf.image.random_contrast(image, 0.5, 1.8)  # Make contrast lower or higher
        return image, label

    @staticmethod
    def augment_strong(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.5)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        return image, label

    @staticmethod
    def add_noise(image, label):
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=random.randint(0, 5)/10)
        if not random.randint(0, 4):
            image = tf.add(image, noise)
        return image, label

    @staticmethod
    def add_noiseandaugment(image, label):
        image, _ = ImageDataset.augment(image, label)
        image, _ = ImageDataset.add_noise(image, label)
        return image, label

    @staticmethod
    def augment_from_param(dataset, param_list):
        # if "augment" in param_list:
        #     return_dataset = dataset.map(ImageDataset.augment)
        # elif "augment_strong" in param_list:
        #     return_dataset = dataset.map(ImageDataset.augment_strong)
        # else:
        #     return_dataset = dataset
        if "noise" and "augment" in param_list:
            return_dataset = dataset.map(ImageDataset.add_noiseandaugment)
        return return_dataset

    def calc_weights(self):
        label_list = []
        for elem in self.dataset_train:
            label_list.append(elem[1].numpy())
        a = sum(label_list)
        b = len(label_list) - sum(label_list)
        class_weights = {0: (1 / b) * (a + b) / 2, 1: (1 / a) * (a + b) / 2}
        logging.info(f"Got {len(label_list)} Training Samples of which {a} are diabetic and {b} are healthy")
        logging.info(f"Class Weights: {class_weights}")
