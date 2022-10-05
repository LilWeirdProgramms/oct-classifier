import matplotlib.pyplot as plt
import numpy as np
import skimage.io as sk_io
import skimage.filters as sk_fi
import tensorflow as tf

from hyperparameterStudy.image_dataset import ImageDataset
from PreprocessData import PreprocessData
from numpy.fft import fft, ifft


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

    # Preprocess and Save:
    def preprocess_dataset(self, file_name, label):
        image, label = self.file_name_to_image(file_name, label)
        image = self._preprocess_image(image.numpy())
        return image, label

    def file_name_to_image(self, filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.io.decode_png(image_string, channels=self.channels, dtype=tf.uint8)
        return image, label

    def _preprocess_image(self, image):
        if self.crop:
            image = image[self.crop:-self.crop, self.crop:-self.crop]
        # image = self.remove_periodic_noise(image, fft_filter=True)
        #image = image[::4, ::4]
        image = tf.image.per_image_standardization(image).numpy()
        if self.channels == 3:
            image = (image - image.min()) / (image.max()-image.min()) * 255
            image = tf.keras.applications.vgg16.preprocess_input(image)
        return image

    def remove_periodic_noise(self, image: np.ndarray, interpolation=False, low_pass=True, fft_filter=False):
        """
        Removes Noise that Appears every 0.25 columns of the image. Works best if image.shape[1] is dividable by 4
        :param image: 2d or 3d array with frequency noise along the First Dimension
        :param interpolation: Interpolate the filterd frequencies or just set them to zero
        :param low_pass: Apply Gauss Filter afterwards
        :return:
        """
        if fft_filter:
            fft_im = np.fft.fft(image, axis=1)
            central_frequency = int(image.shape[1] / 2)
            lower_harmonic = int(central_frequency / 2)
            upper_harmonic = int(3 * central_frequency / 2)
            if interpolation:
                fft_im = self.interpolate_freq(fft_im, central_frequency)
                fft_im = self.interpolate_freq(fft_im, lower_harmonic)
                fft_im = self.interpolate_freq(fft_im, upper_harmonic)
            else:
                fft_im = self.set_freq_zero(fft_im, central_frequency)
                fft_im = self.set_freq_zero(fft_im, lower_harmonic)
                fft_im = self.set_freq_zero(fft_im, upper_harmonic)
            new_im = np.fft.ifft(fft_im, axis=1).real
        else:
            new_im = image
        if low_pass:
            new_im = sk_fi.gaussian(new_im, sigma=1)
        return new_im

    def interpolate_freq(self, freq_data, freq):
        freq_data[:, freq - 1:freq + 2] = np.outer((freq_data[:, freq - 2] + freq_data[:, freq + 3]) / 2, np.ones(3, ))
        return freq_data

    def set_freq_zero(self, freq_data, freq):
        freq_data[:, freq - 1:freq + 2] = 0
        return freq_data

    def save_preprocessed_dataset(self, save_path, data):
        np.save(save_path, data.astype("float32"))

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
            image = tf.image.random_brightness(image, max_delta=0.15)
            image = tf.image.random_contrast(image, 0.8, 1.1)
        # image = tf.image.per_image_standardization(image)
        return image, label








if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    with tf.device("/cpu:0"):
        file_list = ImageDataset.load_file_list("test")
        pid = PreprocessImageData(file_list, rgb=True)
        im, la = pid.preprocess_dataset(*file_list[0])
        print(im.shape)

