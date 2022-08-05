import os
import tensorflow as tf
import logging
import numpy as np
from tensorflow.keras import mixed_precision

from PreprocessData import PreprocessData
from TFRecordsHandler import write_images_to_tfr_short, parse_tfr_element
from ParseBinary import BinaryMILDataset
import InputListUtils


class PreprocessRawData:

    def __init__(self, input_file_list, rgb=False, crop=False, normalize=True, augment=True, data_type="test"):
        """

        :param input_file_list: -> List of raw, or retina or... from whereever (e.g. from server)
        :param rgb:
        :param crop:
        :param normalize:
        :param augment:
        :param data_type:
        """
        self._input_file_list = input_file_list

        self._buffer_folder = self.buffer_folder_generator()
        self.buffer_folder_list = [f"/media/julius/My Passport1/bufferhdd1/{data_type}",
                                   f"/media/julius/My Passport/bufferhdd2/{data_type}",
                                   f"/mnt/NewHDD/bufferhdd0/{data_type}"]
        # self.buffer_folder_list = [f"/media/julius/My Passport1/bufferhdd2/{data_type}/1",
        #                            f"/media/julius/My Passport/bufferhdd1/{data_type}/1",
        #                            f"/mnt/NewHDD/bufferhdd0/{data_type}/1",
        #                            f"/media/julius/My Passport1/bufferhdd2/{data_type}/2",
        #                            f"/media/julius/My Passport/bufferhdd1/{data_type}/2",
        #                            f"/mnt/NewHDD/bufferhdd0/{data_type}/2"]
        self.buffer_cntr = 0

        self.data_type = data_type
        self.calculation_file_list = None
        self.dataset = None
        self.rgb = rgb
        self.crop = crop
        self.normalize = normalize
        self.augment = augment
        self.train_label_list = None

    def buffer_folder_generator(self):
        while True:
            buffer_folder = self.buffer_folder_list[self.buffer_cntr]
            self.buffer_cntr += 1
            if self.buffer_cntr == len(self.buffer_folder_list):
                self.buffer_cntr = 0
            yield buffer_folder

    def get_all_tfrecords(self):
        files_full_path = []
        file_labels = []
        for i in range(len(self.buffer_folder_list)):
            folder = next(self._buffer_folder)
            files = os.listdir(folder)
            files_full_path.extend([os.path.join(folder, file) for file in files])
            file_labels.extend([int(file.split("_")[0]) for file in files])

        # TODO: This needs to sort such that first come the healthy or diabetic than the other
        self.calculation_file_list = sorted(zip(files_full_path, file_labels), key=lambda file: (
            int(os.path.basename(file[0]).split("_")[0]),
            int(os.path.basename(file[0]).split("_")[-2]),
            int(os.path.basename(file[0]).split("_")[-1].split(".")[0])))
        file_labels = [label for file, label in self.calculation_file_list]
        files = [file for file, label in self.calculation_file_list]
        self.train_label_list = file_labels[int(len(file_labels) * 0.075):-int(len(file_labels) * 0.075)]
        return files, file_labels

    def create_dataset_for_calculation(self):
        files_full_path, file_labels = self.get_all_tfrecords()
        dataset = tf.data.Dataset.from_tensor_slices(files_full_path)
        if self.data_type == "train":
            dataset_1 = dataset.take(int(len(files_full_path) * 0.075))
            train_dataset = dataset.skip(int(len(files_full_path) * 0.075)).take(int(len(files_full_path) * 0.85))
            dataset_2 = dataset.skip(int(len(files_full_path) * 0.925)).take(int(len(files_full_path) * 0.075))
            val_dataset = dataset_1.concatenate(dataset_2)
            train_dataset = train_dataset.shuffle(int(len(files_full_path)*0.85))
            train_dataset = train_dataset.interleave(lambda x: (tf.data.TFRecordDataset(x)),
                                         num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
            train_dataset = train_dataset.map(parse_tfr_element)
            val_dataset = val_dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                                         num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
            val_dataset = val_dataset.map(parse_tfr_element)
            return train_dataset, val_dataset
        dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                                             num_parallel_calls=tf.data.AUTOTUNE,
                                             deterministic=True).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(parse_tfr_element)
        return dataset

    def preprocess_data_and_save(self):
        self.delete_all_old()
        for file_name, label in self._input_file_list:
            try:
                save_folder = next(self._buffer_folder)
                raw_data_stack, label = self.preprocess_dataset(file_name, label)
                for i, raw_data in enumerate(raw_data_stack):
                    new_file_name = f"{label}_{os.path.basename(file_name)[:-4]}_{i}"
                    new_file_path = os.path.join(save_folder, new_file_name)
                    self.save_preprocessed_dataset(new_file_path, raw_data, label)
            except BaseException as err:
                print(f"{err} just happend. Skipping Sample")

    def preprocess_dataset(self, file_name, label):
        reader = BinaryMILDataset(reading_type="difference", background_sub=False)
        instance = reader.instance_from_binaries_generator(file_name)
        return instance, label

    def save_preprocessed_dataset(self, save_path, raw_data, label):
        write_images_to_tfr_short(raw_data, filename=save_path, label=label)

    def parse_function(self, data, label):
        raise NotImplementedError("Please Implement this method")

    def augment_function(self, data, label):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def calc_weights(label_list: list):
        a = sum(label_list)
        b = len(label_list) - sum(label_list)
        class_weights = {0: (1 / b) * (a + b) / 2, 1: (1 / a) * (a + b) / 2}
        logging.info(f"Got {len(label_list)} Training Samples of which {a} are diabetic and {b} are healthy")
        logging.info(f"Class Weights: {class_weights}")
        return class_weights

    @staticmethod
    def get_test_train_file_lists(type="raw"):
        train_bin_healthy, test_bin_healthy = InputListUtils.find_binaries_test_train(r"^H([0-9]|[0-9][0-9])", 0,
                                                                       location=InputListUtils.server_location, type=type)
        train_bin_diabetic, test_bin_diabetic = InputListUtils.find_binaries_test_train(r"^D([2-9][0-9]|[0-9][0-9][0-9])$", 1,
                                                                         location=InputListUtils.server_location, type=type)
        train_list = train_bin_healthy + train_bin_diabetic
        test_list = test_bin_healthy + test_bin_diabetic
        return train_list, test_list

    def delete_all_old(self):
        for folder in self.buffer_folder_list:
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

def remove_duplicates(path_list):
    only_files = [os.path.basename(path) for path, label in path_list]
    remove_indexes = []
    for duplicate_file in set([x for x in only_files if only_files.count(x) > 1]):
        remove_indexes.append(only_files.index(duplicate_file))
    for index in sorted(remove_indexes, reverse=True):
        del path_list[index]
    only_id = [s.split(".")[-2].split("_")[-1] for s in only_files]
    for duplicate_id in set([x for x in only_id if only_id.count(x) > 1]):
        print(f"Warning: Duplicate ID {duplicate_id}. This ID will be overwritten")
    return path_list

if __name__ == "__main__":
    # binary_file_list = [("/mnt/p_Zeiss_Clin/Projects/UWF OCTA/Clinical data/MOON1/H32/rechts/raw_1536x2048x2044x2_4608.bin", 1)]
    # raw_data_preprocess = PreprocessRawData(input_file_list=binary_file_list)
    train_list, test_list = InputListUtils.get_test_train_file_lists()
    train_list = remove_duplicates(train_list)
    raw_data_preprocess = PreprocessRawData(input_file_list=train_list, data_type="train")
    raw_data_preprocess.preprocess_data_and_save()
    #raw_data_preprocess.create_dataset_for_calculation()
    # raw_data_preprocess = PreprocessRawData(input_file_list=test_list, data_type="test")
    # raw_data_preprocess.preprocess_data_and_save()



