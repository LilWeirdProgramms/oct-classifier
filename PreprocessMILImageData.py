import numpy as np
import skimage.io as sk_io
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from hyperparameterStudy.image_dataset import ImageDataset
from PreprocessImageData import PreprocessImageData


class PreprocessMILImageData(PreprocessImageData):

    def __init__(self, input_file_list=None, rgb=False, crop=False, normalize=True, data_type="test", augment=True):
        super().__init__(input_file_list, data_type=data_type, rgb=rgb, crop=crop, normalize=normalize, augment=augment)
        self.image_size = None

    def preprocess_dataset(self, file_name, label):
        image, label = self.file_name_to_image(file_name, label)
        image = self._preprocess_image(image.numpy())
        #image = image.numpy()
        self.image_size = image.shape
        image = self.create_patches_from_image(image)
        return image, label

    def create_patches_from_image(self, image):
        x_dim = round(np.floor(self.image_size[0]/10))*10
        y_dim = round(np.floor(self.image_size[1]/10))*10
        image = image[:x_dim, :y_dim, :]
        split_in_b = np.stack(np.split(image, 10))
        return np.stack(np.split(split_in_b, 10, axis=2)).reshape((10*10, int(x_dim/10), int(y_dim/10), 1))\
            .astype("float32")

    def preprocess_data_and_save(self):
        self.delete_all_old()
        for file_name, label in self._input_file_list:
            image_stack, label = self.preprocess_dataset(file_name, label)
            for i, image in enumerate(image_stack):
                new_file_name = f"{label}_{os.path.basename(file_name)[:-4]}_{i}.png"
                new_file_path = os.path.join(self._buffer_folder, new_file_name)
                self.save_preprocessed_dataset(new_file_path, image)

    def delete_all_old(self):
        folder = f"data/buffer/{self.data_type}"
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))

    def sort_files(self, files):
        return sorted(files, key=lambda file: (int(round(float(file.split("_")[0]))),
                                               int(file.split("_")[-2]),
                                               int(file.split("_")[-1][:-4])))


if __name__ == "__main__":
    data_type = "test"
    file_list = ImageDataset.load_file_list(data_type)
    pid = PreprocessMILImageData(file_list, rgb=False, crop=False, data_type=data_type)
    #pid.preprocess_data_and_save()
    ds = pid.create_dataset_for_calculation()
    import matplotlib.pyplot as plt
    for image, file in zip(ds.take(300), pid.calculation_file_list):
        print(file)
        #plt.figure()
        #plt.imshow(image[0], "Greys")
        #plt.colorbar()
        #plt.show()
