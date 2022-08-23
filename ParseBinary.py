import numpy as np
import os
from dataclasses import dataclass
import tensorflow as tf
from typing import Generator
import math
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy.fft as fft

class BinaryMILDataset:
    def __init__(self, reading_type="normal", background_sub=False, selection_of_ascans=None):
        """

        :param reading_type: ["normal"], ["every4"], ["difference"]
        """
        self.reading_type = reading_type
        self.ascan_length = 1536
        self.bscan_length = 2048  # 2048
        self.cscan_length = 2044  # 2044
        self.b_division = 10
        self.c_division = 10
        self.a_size = 1536
        self.b_size = int(self.bscan_length / self.b_division)
        self.c_size = int(self.cscan_length / self.c_division)
        self.selection_of_ascans = selection_of_ascans

        self.file_data_type = np.dtype('<u2')

        self.num_read_from_file = int(self.file_data_type.itemsize /
                                      self.file_data_type.itemsize * self.a_size)
        self.background = None
        self.background_sub = background_sub

        self.pca = PCA(n_components=51)
        self.data_mean = None
        self.data_std = None

    def instance_from_binaries_generator(self, binary_file) -> Generator:
        self.calc_preproc_params()
        with open(binary_file, "rb") as f:
            #self.background = self.get_background(f)
            for i in range(self.c_division):
                for j in range(self.b_division):
                    index = self.file_data_type.itemsize * (j * self.b_size +
                                                            2 * i * self.c_size * self.bscan_length) * self.ascan_length
                    f.seek(index, os.SEEK_SET)
                    a = self._create_instance(f)
                    yield a

    def _create_instance(self, file):
        """
        Employs constant Padding
        Move to: tf.data.TFRecordDataset(filenames = [fsns_test_file])
        1 for greyscale image
        """
        instance = np.empty((self.c_size, int(self.b_size / 4), int(self.b_size / 4), 1), "float32")
        for c_index in range(self.c_size):
            if self.reading_type == "difference":
                read_bscan =self.get_difference_bscans(file)
            else:
                read_bscan = self.get_bscans(file)
            read_bscan = read_bscan[::4, :]
            preproc_bscan = self.preproc_layer1(read_bscan)
            preproc_bscan = self.preproc_layer2(preproc_bscan)
            instance[c_index, :, :, 0] = preproc_bscan
        #if self.reading_type == "every4":
        instance = instance[::4, :, :, :]

        #if self.background_sub:
        #    instance = self.substract_background(instance)
        return instance   # TODO: Normalize Instance Bag wise (moving average?)

    def calc_preproc_params(self):
        prepared_ascans = self.preproc_layer1(self.selection_of_ascans)
        self.pca.fit(prepared_ascans)
        reduced_data = self.pca.transform(prepared_ascans)
        self.data_mean = reduced_data.mean()
        self.data_std = reduced_data.std()

    def preproc_layer1(self, data: np.ndarray):
        """
        Input a list of raw a-scan
        :param data:
        :return:
        """
        # crop:
        data_transformed = data[:, 400:1400]
        # maybe do fourier transform:
        data_transformed = np.abs(fft.ifft(data_transformed))[:, 1:]
        return data_transformed

    def preproc_layer2(self, data):
        """
        In this layer all the preprocessing is done that relies on information from the whole dataset
        :param data:
        :return: (51, 51, 51)
        """
        reduced_data = self.pca.transform(data)
        reduced_normalize_data = (reduced_data - self.data_mean) / self.data_std
        return reduced_normalize_data

    def get_difference_bscans(self, file):
        bscan = np.empty((self.b_size, self.a_size, 2))
        for i in range(2):
            for b_index in range(self.b_size):
                read_from_file = np.fromfile(file, dtype=self.file_data_type, count=self.num_read_from_file)
                bscan[b_index, :, i] = read_from_file / 65535
            file.seek(self.file_data_type.itemsize *
                      self.ascan_length * (1 * self.bscan_length - self.b_size)
                      , os.SEEK_CUR)
        bscan = bscan.astype("float32")
        bscan = bscan[..., 0] - bscan[..., 1]
        return bscan

    def get_bscans(self, file):
        bscan = np.empty((self.b_size, self.a_size))
        for b_index in range(self.b_size):
            read_from_file = np.fromfile(file, dtype=self.file_data_type, count=self.num_read_from_file)
            bscan[b_index, :] = read_from_file
        file.seek(self.file_data_type.itemsize *
                  self.ascan_length * (2 * self.bscan_length - self.b_size)
                  , os.SEEK_CUR)
        bscan = bscan.astype("float32")
        return bscan

    def substract_background(self, instance):
        for i in range(instance.shape[0]):
            instance[i, :, :, 0] = instance[i, :, :, 0] - self.background[::4]
        # instance - self.background[None, ..., None]
        return instance

    def get_background(self, file):
        file.seek(1536*2043*2048*2*2)
        background = np.fromfile(file, dtype="uint16", count=1536*2048)[:1536*self.b_size].reshape((self.b_size, 1536)) \
                     / 65535
        if self.reading_type == "difference":
            background_diff = np.fromfile(file, dtype="uint16", count=1536 * 2048)[:1536*self.b_size]\
                .reshape((self.b_size, 1536)) / 65535
            background = background.astype("float32")
            background_diff = background_diff.astype("float32")
            background = background - background_diff
        return background

    def _check_file_existance(self, file_list: list):
        file_exists = [not os.path.exists(file) for file, label in file_list]
        if any(file_exists):
            missing_file = file_list[np.argmax(file_exists)]
            raise FileExistsError(f"The File {missing_file} was not found on disk")

if __name__ == "__main__":
    binary_file = "/mnt/p_Zeiss_Clin/Projects/UWF OCTA/Clinical data/MOON1/H32/rechts/raw_1536x2048x2044x2_4608.bin"
    reader = BinaryMILDataset(reading_type="difference", background_sub=True)
    for a in reader.instance_from_binaries_generator(binary_file):
        plt.figure()
        plt.plot(a[196, 36, :, 0])
        plt.show()
        break
