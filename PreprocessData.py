import os
import tensorflow as tf
import logging


class PreprocessData:

    def __init__(self, input_file_list=None, data_type="test"):
        self._input_file_list = input_file_list
        self._buffer_folder = f"data/buffer/{data_type}"
        self.data_type = data_type
        self.calculation_file_list = None
        self.dataset = None

    def preprocess_data_and_save(self):
        for file_name, label in self._input_file_list:
            image, label = self.preprocess_dataset(file_name, label)
            new_file_name = f"{label}_{os.path.basename(file_name)}"
            new_file_path = os.path.join(self._buffer_folder, new_file_name)
            self.save_preprocessed_dataset(new_file_path, image)

    def create_dataset_for_calculation(self):
        files = os.listdir(self._buffer_folder)
        files = self.sort_files(files)
        files_full_path = [os.path.join(self._buffer_folder, file) for file in files]
        file_labels = [int(file.split("_")[0]) for file in files]
        self.calculation_file_list = [(file, label) for file, label in zip(files_full_path, file_labels)]
        dataset = tf.data.Dataset.from_tensor_slices((files_full_path, file_labels))
        if self.data_type == "train":
            dataset_1 = dataset.take(int(len(files_full_path) * 0.075))
            train_dataset = dataset.skip(int(len(files_full_path) * 0.075)).take(int(len(files_full_path) * 0.85))
            dataset_2 = dataset.skip(int(len(files_full_path) * 0.925)).take(int(len(files_full_path) * 0.075))
            val_dataset = dataset_1.concatenate(dataset_2)
            train_dataset = train_dataset.shuffle(int(len(files_full_path)*0.85))
            train_dataset = train_dataset.map(self.parse_function)
            train_dataset = train_dataset.map(self.augment_function)
            val_dataset = val_dataset.map(self.parse_function)
            return train_dataset, val_dataset
        dataset = dataset.map(self.parse_function)
        return dataset

    def sort_files(self, files):
        return files
        #return sorted(files, key=lambda file: (int(file.split("_")[-2]), int(file.split("_")[-1][:-4])))
    # TODO: return files

    def preprocess_dataset(self, file_name, label):
        raise NotImplementedError("Please Implement this method")

    def save_preprocessed_dataset(self, save_path, image):
        raise NotImplementedError("Please Implement this method")

    def parse_function(self, data, label):
        raise NotImplementedError("Please Implement this method")

    def augment_function(self, data, label):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def load_file_list(test_or_train="test", angio_or_structure="images"):
        """

        :param test_or_train: one of ["test", "train"]
        :param angio_or_structure: one of ["images" (angio), "structure", "combined"]
        :return:
        """
        # TODO: Label Smoothing
        input_file_list = PreprocessData.input_list_from_folder(f"data/diabetic_{angio_or_structure}/{test_or_train}_files", 1) \
                          + PreprocessData.input_list_from_folder(f"data/healthy_{angio_or_structure}/{test_or_train}_files", 0)
        return input_file_list

    @staticmethod
    def input_list_from_folder(folder, label):
        files = os.listdir(folder)
        input_file_list = [(os.path.join(folder, file), label) for file in files]
        return input_file_list

    @staticmethod
    def calc_weights(ds):
        label_list = []
        for elem in ds:
            label_list.append(round(elem[1].numpy()))
        a = sum(label_list)
        b = len(label_list) - sum(label_list)
        class_weights = {0: (1 / b) * (a + b) / 2, 1: (1 / a) * (a + b) / 2}
        logging.info(f"Got {len(label_list)} Training Samples of which {a} are diabetic and {b} are healthy")
        logging.info(f"Class Weights: {class_weights}")

