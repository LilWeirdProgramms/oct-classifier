import tensorflow as tf
import Callbacks
import image2d
import InputListUtils
import logging
import random
import numpy as np
import skimage.io as sk_io
import skimage.color as sk_co
import os

class ImageDataset():
    def __init__(self, data_list=None, validation_split=True, mil=False, rgb=False):
        if data_list is None:
            self.data_list = ImageDataset.get_file_list()
        else:
            self.data_list = data_list
            if validation_split:
                random.shuffle(self.data_list)
            #random.shuffle(self.data_list)
        self.mil = mil
        self.validation_split = validation_split
        self.dataset_train: tf.data.Dataset = None
        self.dataset_val: tf.data.Dataset = None
        self.mil_list = []
        self.rgb = rgb
        self.create_dataset_from_file_list()
        #self.train_data: np.ndarray = None

    @staticmethod
    def load_file_list(test_or_train="test", angio_or_structure="images"):
        input_file_list = ImageDataset.input_list_from_folder(f"data/diabetic_{angio_or_structure}/{test_or_train}_files", 1) \
                          + ImageDataset.input_list_from_folder(f"data/healthy_{angio_or_structure}/{test_or_train}_files", 0)
        return input_file_list

    @staticmethod
    def input_list_from_folder(folder, label):
        files = os.listdir(folder)
        input_file_list = [(os.path.join(folder, file), label) for file in files]
        return input_file_list

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
        if self.mil:
            self.create_mil_dataset()
        else:
            self.create_supervised_dataset()

    def create_mil_dataset(self):
        all_image_patches = None
        all_labels = None
        for path, label in self.data_list:
            image = sk_io.imread(path, as_gray=True)
            cropat_b, cropat_c = 300 + 22, 300 + 24
            image = image[cropat_b:-cropat_b, cropat_c:-cropat_c]
            image = (image - image.mean()) / image.std()
            image_patches = ImageDataset.create_patches_from_image(image)
            if all_image_patches is None:
                all_image_patches = image_patches
                all_labels = np.zeros(100).astype("float32") + label
            else:
                all_image_patches = np.concatenate([all_image_patches, image_patches])
                all_labels = np.concatenate([all_labels, np.zeros(100) + label])
            self.mil_list.extend([path]*100)
            print(f"Preprocessed {all_labels.size / 100} of {len(self.data_list)}")
        if self.validation_split:
            train_data = all_image_patches[:-3000]
            train_labels = all_labels[:-3000]
            val_data = all_image_patches[-3000:]
            val_labels = all_labels[-3000:]
            train_paths = self.mil_list[:-3000]
            val_paths = self.mil_list[-3000:]
            train_data, train_labels, train_paths = self.shuffle_mil_dataset(train_data, train_labels, train_paths)
        else:
            self.train_data, train_labels, train_paths = all_image_patches, all_labels, self.mil_list
        self.dataset_train = tf.data.Dataset.from_tensor_slices((self.train_data, train_labels))
        if self.validation_split:
            self.dataset_val = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
            self.mil_list = train_paths + val_paths
        else:
            self.mil_list = train_paths

    def shuffle_mil_dataset(self, patches, labels, paths):
        data_tuple = list(zip(patches, labels, paths))
        random.shuffle(data_tuple)
        patches, labels, paths = zip(*data_tuple)
        return np.array(patches), np.array(labels), list(paths)

    @staticmethod
    def create_patches_from_image(image):
        split_in_b = np.stack(np.split(image, 10))
        return np.stack(np.split(split_in_b, 10, axis=2)).reshape((100, 140, 140, 1)).astype("float32")

    def create_supervised_dataset(self):
        x_train = []
        y_train = []
        id = image2d.ImageDataset()
        for path, label in self.data_list:
            image, _ = id._parse_image(path, label)
            if self.rgb:
               image = self.turn_into_rgb(image)
            x_train.append(image)
            y_train.append(label)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.create_dataset_from_array(x_train, y_train)

    def turn_into_rgb(self, image):
        image = sk_co.gray2rgb(np.squeeze(image.numpy()))
        return image

    def create_dataset_from_array(self, data_array, label_array):
        if self.validation_split:
            self.dataset_train = tf.data.Dataset.from_tensor_slices((data_array[:-30], label_array[:-30]))
            self.dataset_val = tf.data.Dataset.from_tensor_slices((data_array[-30:], label_array[-30:]))
        else:
            self.dataset_train = tf.data.Dataset.from_tensor_slices((data_array, label_array))

    @staticmethod
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.7, 1.3)  # Make contrast lower or higher
        return image, label

    @staticmethod
    def augment_strong(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        return image, label

    @staticmethod
    def add_noise(image, label):
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=random.randint(0, 2)/100)
        if not random.randint(0, 10):
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

    @staticmethod
    def calc_weights(ds):
        label_list = []
        for elem in ds:
            label_list.append(elem[1].numpy())
        a = sum(label_list)
        b = len(label_list) - sum(label_list)
        class_weights = {0: (1 / b) * (a + b) / 2, 1: (1 / a) * (a + b) / 2}
        logging.info(f"Got {len(label_list)} Training Samples of which {a} are diabetic and {b} are healthy")
        logging.info(f"Class Weights: {class_weights}")

