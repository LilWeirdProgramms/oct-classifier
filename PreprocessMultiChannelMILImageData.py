import numpy as np
import skimage.filters as sk_fi
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from PreprocessMILImageData import PreprocessMILImageData


class PreprocessMultiChannelMILImageData(PreprocessMILImageData):
    """
    Channel 1 = Angio
    Channel 2 = Structure
    """

    def __init__(self, input_file_list=None, rgb=False, crop=False, normalize=True, data_type="test", augment=True):
        super().__init__(input_file_list, data_type=data_type, rgb=rgb, crop=crop, normalize=normalize, augment=augment)

    def file_name_to_image(self, filenames, label):
        """

        :param filename: tuple of all the channel file names
        :param label:
        :return:
        """
        all_channels = []
        for filename in filenames:
            image_string = tf.io.read_file(filename)
            image_channel = tf.io.decode_png(image_string, channels=self.channels, dtype=tf.uint8)
            all_channels.append(image_channel)
        image = np.concatenate(all_channels, axis=2)
        return image, label

    def preprocess_dataset(self, file_name, label):
        image, label = self.file_name_to_image(file_name, label)
        image = self._preprocess_image(image)
        self.image_size = image.shape
        image = self.create_patches_from_image(image, channels=2)
        return image, label

    def preprocess_data_and_save(self):
        self.delete_all_old()
        for file_name, label in self._input_file_list:
            image_stack, label = self.preprocess_dataset(file_name, label)
            for i, image in enumerate(image_stack):
                new_file_name = f"{label}_{os.path.basename(file_name[0])[:-4]}_{i}"
                new_file_path = os.path.join(self._buffer_folder, new_file_name)
                self.save_preprocessed_dataset(new_file_path, image)

    @staticmethod
    def find_channel_pairs(angio_images, structure_images):
        combined_file_list = []
        for angio_image, label1 in angio_images:
            for structure_image, label2 in structure_images:
                if os.path.basename(angio_image.replace("retina", "")) == \
                        os.path.basename(structure_image.replace("enf", "")):
                    if label2 != label1:
                        print("Labels of channel pairs are different, this means error")
                    combined_file_list.append(((angio_image, structure_image), label1))
        return combined_file_list


if __name__ == "__main__":
    data_type = "test"
    file_list_angio = PreprocessMultiChannelMILImageData.load_file_list(data_type, angio_or_structure="images")
    file_list_struc = PreprocessMultiChannelMILImageData.load_file_list(data_type, angio_or_structure="structure")
    file_list_combined = PreprocessMultiChannelMILImageData.find_channel_pairs(file_list_angio, file_list_struc)
    pid = PreprocessMultiChannelMILImageData(file_list_combined, rgb=False, crop=False, data_type=data_type)
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


