import numpy as np
import skimage.io as sk_io
import skimage.filters as sk_fi
import tensorflow as tf

from hyperparameterStudy.image_dataset import ImageDataset
from PreprocessData import PreprocessData


class PreprocessImageData(PreprocessData):

    def __init__(self, input_file_list=None, rgb=False, crop=False, normalize=True, data_type="test", augment=True):
        super().__init__(input_file_list, data_type)
        if rgb:
            self.channels = 3
        else:
            self.channels = 1
        self.augment = augment
        self.crop = crop
        self.normalize = normalize

    def file_name_to_image(self, filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.io.decode_png(image_string, channels=self.channels, dtype=tf.uint8)
        return image, label

    # Preprocess and Save:
    def preprocess_dataset(self, file_name, label):
        image, label = self.file_name_to_image(file_name, label)
        image = self._preprocess_image(image.numpy())
        return image, label

    # def save_preprocessed_dataset(self, save_path, image):
    #     sk_io.imsave(save_path, image)

    def _preprocess_image(self, image):
        if self.crop:
            image = image[self.crop:-self.crop, self.crop:-self.crop]
        image = sk_fi.rank.mean(image, np.ones((4, 4, 1)))
        image = tf.image.per_image_standardization(image)
        return image

    def save_preprocessed_dataset(self, save_path, data):
        np.save(save_path[:-4], data.astype("float32"))

    # Load, Preprocess and Calc:
    def parse_function(self, filename, label):
        data = tf.py_function(self.parse_numpy, inp=[filename],
            Tout=tf.float32)
        return data, label

    def parse_numpy(self, filename):
        data = np.load(filename.numpy())
        return data

    # Load, Preprocess and Calc:
    # def parse_function(self, image, label):
    #     image, label = self.file_name_to_image(image, label)
    #     image = tf.image.per_image_standardization(image)
    #     return image, label

    def augment_function(self, image, label):
        if self.augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, 0.8, 1.1)
        # image = tf.image.per_image_standardization(image)
        return image, label








if __name__ == "__main__":
    file_list = ImageDataset.load_file_list("test")
    pid = PreprocessImageData(file_list, rgb=True)
    im, la = pid.preprocess_dataset(*file_list[0])
    print(im.shape)

