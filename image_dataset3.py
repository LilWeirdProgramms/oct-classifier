import tensorflow as tf

class ImDataset:

    def __init__(self, rgb=False, folder="data/buffer/test"):
        if rgb:
            self.channels = 3
        else:
            self.channels = 1

    def parse_image_processing(self, filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.io.decode_png(image_string, channels=self.channels)
        return image, label
