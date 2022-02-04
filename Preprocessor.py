import numpy as np
from tensorflow import data as tf_data
from typing import Iterator
from tensorflow.keras.layers.experimental.preprocessing import Normalization


class Preprocessor:

    def __init__(self, dataset: tf_data.Dataset):
        self.dataset = dataset
        self.mean = None
        self.std = None

    def batch(self, batch_size):
        return self.dataset.shuffle(10).batch(batch_size)  # TODO: Without shuffle

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
        normalizer.adapt(self.dataset.map(lambda x, y: x))
        self.mean = normalizer.mean
        self.std = normalizer.variance**0.5
        return normalizer
