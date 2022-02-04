import numpy as np
import os
from dataclasses import dataclass
import tensorflow as tf
from typing import Generator
import math


@dataclass
class InstanceDim:
    asize: int
    bsize: int
    csize: int
    btimes: int
    ctimes: int

# TODO: Dont only accept File List with Labels
class BinaryReader:
    """
    Validation Split using .fit(validation_split=0.2) -> Keras
    Suffling and Batching: fmnist_train_ds.shuffle(5000).batch(32)
    """
    def __init__(self, validation_split=0.2):
        self.ascan_length = 1536  # TODO: Make size explizit
        self.bscan_length = 2047
        self.cscan_length = 2045
        self.validation_split = validation_split
        self.data_type = np.dtype('<u2')
        self.info_map = []

    def create_test_dataset(self, file_list):
        """

        :param file_list: Pass Testing Files List
        :return: Dataset for test purpose
        """
        np.random.shuffle(file_list)
        return self.create_dataset(file_list)

    def create_training_datasets(self, file_list):
        """

        :param file_list: Pass Training File List
        :return: Datasets for training and validation purpose
        """
        self._check_file_existance(file_list)
        np.random.shuffle(file_list)
        train, val = self.split_file_list_for_validation(file_list)
        train_dataset = self.create_dataset(train)
        validation_dataset = self.create_dataset(val)
        return train_dataset, validation_dataset

    def instance_from_binaries_generator(self, list_of_files, evaluate=False) -> Generator:
        """

        :param evaluate: Indicate that the set is generated for evaluation  TODO: Can there be one parameter that is used also in the Preprocessor?
        :param list_of_files:
        :return:
        """
        self.info_map = []
        instance_size = self._decide_instance_size()
        for i in range(instance_size.ctimes):
            for j in range(instance_size.btimes):  # TODO: Make 2 explicit
                for filepath, label in list_of_files:
                    with open(filepath, "rb") as f:  # TODO: trenn die Ausschneidelogik von der lese Logik
                        index = self.data_type.itemsize * (j * instance_size.bsize +
                                                           2 * i * instance_size.csize * self.bscan_length) * \
                                self.ascan_length
                        #if evaluate:
                        #    self._create_info_map(filepath, [i, j])
                        f.seek(index, os.SEEK_SET)
                        yield self._create_instance(f, instance_size), int(label)

    def create_dataset(self, file_list) -> tf.data.Dataset:
        """
        balanced_ds = tf.data.Dataset.sample_from_datasets([negative_ds, positive_ds], [0.5, 0.5]).
        :return:
        """
        instance_size = self._decide_instance_size()
        dataset = tf.data.Dataset.from_generator(
            self.instance_from_binaries_generator, args=[[(data, str(label)) for data, label in file_list]],
            output_signature=(tf.TensorSpec(shape=(self.ascan_length, instance_size.bsize, instance_size.csize, 1),
                                            dtype=self.data_type),
                              tf.TensorSpec(shape=(), dtype=np.dtype('u1')))
            ).prefetch(tf.data.AUTOTUNE)
        return dataset

    def split_file_list_for_validation(self, file_list):
        """
        Shuffle first!
        :param file_list:
        :return:
        """
        split_at = self._one_or_80_percent(file_list)
        training_files = file_list[:split_at]
        validation_files = file_list[split_at:]
        got_healthy = any([elem[1] == 0 for elem in validation_files])
        got_diabetic = any([elem[1] == 1 for elem in validation_files])
        if got_healthy and got_diabetic:
            print("Warning: Only files of one Dataset are present in the Validation Dataset")
        return training_files, validation_files

    def _one_or_80_percent(self, file_list):
        return max(math.floor(len(file_list)*(1-self.validation_split)), 1)

    def _create_instance(self, file, instance_size: InstanceDim):
        """
        Move to: tf.data.TFRecordDataset(filenames = [fsns_test_file])
        1 for greyscale image
        """
        instance = np.empty((self.ascan_length, instance_size.bsize, instance_size.csize, 1), self.data_type)
        for c_index in range(instance_size.csize):
            for b_index in range(instance_size.bsize):  # TODO: Make 2 explicit
                instance[:, b_index, c_index, 0] = np.fromfile(file, dtype=self.data_type, count=self.ascan_length)  # TODO: MAke sure it also wprks with binary
            file.seek(self.data_type.itemsize *
                      self.ascan_length*(2 * self.bscan_length - instance_size.bsize)
                      , os.SEEK_CUR)
        return instance

    def _create_info_map(self, bag_name_path: str, instance_position):
        """
        Creates an info map "on the fly" while the generator is iterated
        :param bag_name_path: File Path of Bag
        :param instance_position: Position of Instance in the File; remains to be seen where the indeces start
        :return:
        """
        file_name = bag_name_path.split(os.sep)
        number = len(self.info_map)
        self.info_map.append([number, instance_position, file_name])

    def _decide_instance_size(self):
        """
        Bag hat Instanzen
        """
        # for dim in [self.Bscan_length, self.Ascan_length]:
        #     for i in range(1,dim):
        #         if dim%i < 2:
        #             print(i, dim//i, dim%i) -> Slice 89, 73 mal
        self.cscan_length = 2044  # -> One has to go
        return InstanceDim(1536, 23, 28, 89, 73)

    def _check_file_existance(self, file_list: list):  # TODO: File list refactor
        file_exists = [not os.path.exists(file) for file, label in file_list]
        if any(file_exists):
            missing_file = file_list[np.argmax(file_exists)]
            raise FileExistsError(f"The File {missing_file} was not found on disk")



