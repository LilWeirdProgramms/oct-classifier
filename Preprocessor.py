from tensorflow import data as tf_data


class Preprocessor:

    def __init__(self):
        self.batch_size = 20

    def preprocess(self, input_data: tf_data.Dataset):
        dataset = self.shuffle_batch(input_data)
        return dataset

    def shuffle_batch(self, input_data: tf_data.Dataset):
        return input_data.shuffle(input_data.cardinality().numpy()).batch(self.batch_size)


