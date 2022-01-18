from tensorflow import data as tf_data

class Preprocessor:

    def __init__(self, input_data: tf_data.Dataset):
        self.data = input_data

    def normalize_and_zero_center(self):
        #return self.data.map()
        pass

    def batch(self):
        pass