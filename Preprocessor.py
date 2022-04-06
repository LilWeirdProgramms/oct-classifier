import numpy as np
from tensorflow import data as tf_data
from typing import Iterator
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import os

class Preprocessor:

    def __init__(self, dataset: tf_data.Dataset):
        self.dataset = dataset
        self.mean = None
        self.std = None

    def batch(self, batch_size):
        return self.dataset.batch(batch_size)  # TODO: Without shuffle

    def normalize_dataset(self):
        self.calc_moments()
        self.dataset = self.dataset.map(lambda x, y: ((x.numpy() - self.mean) / self.std, y))

    def calc_moments(self):
        """
        Calcs first and second Moments (Mean and Standard Deviation) of current Dataset
        """
        print("Calculating Mean and Standard Deviation of whole Dataset. This could take a while")
        self.mean, self.std, n, old_std, old_mean = 0, 0, 0, 0, 0
        for element in self.dataset:
            self.mean += np.mean(element[0])
            self.std += np.std(element[0])
            if self.std == old_std or self.mean == old_mean:
                print("Warning: Probably a float overflow at n=" + str(n))
            old_std, old_mean = self.std, self.mean
            n += 1
        self.mean /= n
        self.std /= n
        print("Mean: " + str(self.mean) + "; Standard Deviation: " + str(self.std))

    def normalize_layer(self):
        normalizer = Normalization(axis=None)
        normalizer.adapt(self.dataset.map(lambda x, y: x).take(10))
        self.mean = normalizer.mean
        self.std = normalizer.variance**0.5
        return normalizer

    def get_boundary_files(self, data_location):
        data_location = "/mnt/NewHDD/tfrecords"
        boundary_files = []
        all_labels = []
        for instance in os.listdir(data_location):
            label = int(instance[0])
            instance = os.path.join(data_location, instance)
            for file in os.listdir(instance):
                is_corner_instance, corner_label = instance_condition(file, label)
                if is_corner_instance:
                    boundary_files.append(os.path.join(instance, file))
                    all_labels.append(corner_label)
        return boundary_files

    def is_boundary_instance(self, file_name, instance_label, bag_size):
        position_string = file_name.split(".")[0]
        b_pos_str, c_pos_str = position_string.split("_")
        b_pos = int(b_pos_str[1:])
        c_pos = int(c_pos_str[1:])
        instance_label *= 2
        if b_pos == 0 and c_pos == 0:
            is_corner = True
        elif b_pos == bag_size[0] - 1 and c_pos == bag_size[1] - 1:
            is_corner = True
            instance_label += 1
        else:
            is_corner = False
        return is_corner, instance_label

