import numpy as np
from tensorflow import data as tf_data
from typing import Iterator
from tensorflow.keras.layers.experimental.preprocessing import Normalization

class Preprocessor:

    def __init__(self, dataset: tf_data.Dataset):
        self.batch_size = 20
        self.dataset = dataset

    def batch(self):
        return self.dataset.shuffle(89*73).batch(self.batch_size)  # TODO:

    def standardize(self, data_generator: Iterator[tuple]):
        mean, var, n = 0, 0, 0
        for element in data_generator:
            mean += np.mean(element[0])
            var += np.var(element[0])
            n += 1
        mean /= n
        var /= n
        return

    def normalize(self):
        normalizer = Normalization()
        normalizer.adapt(self.dataset)
        return normalizer





