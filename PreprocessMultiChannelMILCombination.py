import numpy as np
import skimage.filters as sk_fi
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from PreprocessMultiChannelMILImageData import PreprocessMultiChannelMILImageData
from PreprocessData import PreprocessData


class PreprocessMultiChannelMILCombination(PreprocessMultiChannelMILImageData):

    def __init__(self, input_file_list=None, rgb=False, crop=False, normalize=True, data_type="test", augment=True):
        super().__init__(input_file_list, data_type=data_type, rgb=rgb, crop=crop, normalize=normalize, augment=augment)

    def _preprocess_image(self, image):
        if self.crop:
            image = image[self.crop:-self.crop, self.crop:-self.crop]
        image = self.remove_periodic_noise(image)
        return image

    def preprocess_dataset(self, file_name, label):
        image, label = self.file_name_to_image(file_name, label)
        image = self._preprocess_image(image)
        self.image_size = image.shape
        image = self.combine_images(image)
        image = self.create_patches_from_image(image, channels=1)
        return image, label

    def combine_images(self, both_image_channels):
        angio_im = both_image_channels[..., 0]
        struct_im = both_image_channels[..., 1]
        angio_im = (angio_im - angio_im.mean()) / angio_im.std()
        struct_im = (struct_im - struct_im.min()) / struct_im.max()
        combined_im = np.multiply(angio_im, struct_im)
        combined_im = (combined_im - combined_im.mean()) / combined_im.std()
        combined_im = np.expand_dims(combined_im, axis=2)
        return combined_im

if __name__ == "__main__":
    data_type = "test"
    file_list_angio = PreprocessData.load_file_list(data_type, angio_or_structure="images")
    file_list_struc = PreprocessData.load_file_list(data_type, angio_or_structure="structure")
    file_list_combined = PreprocessMultiChannelMILImageData.find_channel_pairs(file_list_angio, file_list_struc)
    pid = PreprocessMultiChannelMILCombination(file_list_combined, rgb=False, crop=False, data_type=data_type)
    pid.preprocess_data_and_save()
    ds = pid.create_dataset_for_calculation()
    # import matplotlib.pyplot as plt
    # for image, file in zip(ds.take(300), pid.calculation_file_list):
    #     print(file)
    #     #plt.figure()
    #     #plt.imshow(image[0], "Greys")
    #     #plt.colorbar()
    #     #plt.show()
    i = 1

