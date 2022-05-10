import numpy as np
import os
import logging
import math
import tensorflow as tf


class DatasetBaseClass:

    def __init__(self, val_split=0.2):
        self.validation_split = val_split
        self._training_files = None
        self._test_files = None

    @property
    def training_files(self):
        if self._training_files is None:
            self._training_files = self.get_training_files()
        return self._training_files

    @property
    def test_files(self):
        if self._test_files is None:
            self._test_files = self.get_test_files()
        return self._test_files

    def create_dataset(self, files, deterministic=True, augmented=False):
        raise NotImplementedError("Implement this function in derived class")

    def get_training_files(self):
        raise NotImplementedError("Implement this function in derived class")

    def get_test_files(self):
        raise NotImplementedError("Implement this function in derived class")

    def get_training_datasets(self):
        """
        Takes training_files from InputList, splits into training and validation and returns the two datasets
        :return: Training Dataset consisting of [Image, label] and Validation Dataset
        """
        train_files, val_files = self.split_file_list_for_validation(self.training_files)
        train_ds = self.create_dataset(train_files, deterministic=False)
        val_ds = self.create_dataset(val_files, augmented=False)
        return train_ds, val_ds, train_files, val_files

    def get_test_datasets(self):
        return self.create_dataset(self.test_files)

    def split_file_list_for_validation(self, file_list):
        """
        Shuffle first!
        :param file_list:
        :return:
        """
        np.random.shuffle(file_list)
        split_at = self._one_or_80_percent(file_list)
        training_files = file_list[:split_at]
        validation_files = file_list[split_at:]
        logging.info(f"Current Validation Files: \n {validation_files}")
        return training_files, validation_files

    def _one_or_80_percent(self, file_list):
        return max(math.floor(len(file_list)*(1-self.validation_split)), 1)

    def remove_invalid_file_paths(self, file_list, image=True):
        existing_file_list = []
        for file_path in file_list:
            if os.path.exists(file_path[0]):
                use_file = True
                if image:
                    use_file = self.remove_invalid_image(file_path[0])
                if use_file:
                    existing_file_list.append(file_path)
                else:
                    logging.warning(f"File {file_path[0]} has corrupt dimensions")
            else:
                logging.warning(f"File {file_path[0]} not found and removed from Input List")
        return existing_file_list

    def remove_invalid_image(self, file):
        image_string = tf.io.read_file(file)
        image = tf.io.decode_png(image_string, channels=1)
        return image.shape == (2044, 2048, 1)

