from tensorflow import data as tf_data


class Preprocessor:

    def __init__(self):
        self.batch_size = 20

    def preprocess(self, input_data: tf_data.Dataset):
        dataset = self.batch(input_data)
        return dataset

    def batch(self, input_data: tf_data.Dataset):
        return input_data.shuffle(89*73).batch(self.batch_size)  # TODO:


