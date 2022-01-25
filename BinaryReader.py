import InputList
import numpy as np
import os
from dataclasses import dataclass
import tensorflow as tf
from typing import Iterator

@dataclass
class InstanceDim:
    bsize: int
    csize: int
    btimes: int
    ctimes: int

"""
Validation Split using .fit(validation_split=0.2) -> Keras
Suffling and Batching: fmnist_train_ds.shuffle(5000).batch(32)
"""
class BinaryReader:

    def __init__(self):
        self.ascan_length = 1536
        self.bscan_length = 2047
        self.cscan_length = 2045
        self.data_type = np.dtype('<u2')

    def instance_from_binaries_generator(self, list_of_files, label: int) -> Iterator: #TODO:
        """

        :param list_of_files:
        :param label: Label of Bag -- 1 is pathology 0 is healthy
        :return:
        """
        instance_size = self._decide_instance_size()
        for filepath in list_of_files:
            with open(filepath, "rb") as f:
                for i in range(instance_size.ctimes):
                    for j in range(instance_size.btimes):
                        f.seek(self.data_type.itemsize*j*instance_size.bsize*self.ascan_length, os.SEEK_SET)
                        yield self._create_instance(f, instance_size), label

    def create_training_datasets(self) -> tf.data.Dataset:
        """
        balanced_ds = tf.data.Dataset.sample_from_datasets([negative_ds, positive_ds], [0.5, 0.5]).
        :return:
        """
        instance_size = self._decide_instance_size()
        diabetic_dataset = tf.data.Dataset.from_generator(
            self.instance_from_binaries_generator, args=[InputList.diabetic_training_files, 1],
            output_signature=(tf.TensorSpec(shape=(self.ascan_length, instance_size.bsize, instance_size.csize),
                                            dtype=self.data_type),
                              tf.TensorSpec(shape=(), dtype=np.dtype('u1')))
            ).prefetch(1)
        # diabetic_dataset = tf.data.Dataset.from_generator(
        #     self.instance_from_binaries_generator, args=[InputList.diabetic_training_files, 0]
        # ).prefetch(1)
        # training_dataset = tf.data.Dataset.zip((healthy_dataset, diabetic_dataset))
        training_dataset = diabetic_dataset
        return training_dataset

    """
    Move to: tf.data.TFRecordDataset(filenames = [fsns_test_file])
    """
    def _create_instance(self, file, instance_size: InstanceDim):
        instance = np.empty((self.ascan_length, instance_size.bsize, instance_size.csize), self.data_type)
        for c_index in range(instance_size.csize):
            for b_index in range(instance_size.bsize):
                instance[:, b_index, c_index] = np.fromfile(file, dtype=self.data_type, count=self.ascan_length) #TODO: MAke sure it also wprks with binary
            file.seek(self.data_type.itemsize*self.ascan_length*(self.bscan_length-instance_size.bsize), os.SEEK_CUR)
        return instance

    """
    Bag hat Instanzen
    """
    def _decide_instance_size(self):
        # for dim in [self.Bscan_length, self.Ascan_length]:
        #     for i in range(1,dim):
        #         if dim%i < 2:
        #             print(i, dim//i, dim%i) -> Slice 89, 73 mal
        self.cscan_length = 2044 # -> One has to go
        return InstanceDim(23, 28, 89, 73)

    def get_test_samples(self):
        pass


# Batches from one File??
# -> Can I Make test hdf5 Dataset?

hallo = BinaryReader()
