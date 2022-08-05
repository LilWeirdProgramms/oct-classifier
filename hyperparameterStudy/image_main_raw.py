import keras.backend
import matplotlib.pyplot as plt
import random
import itertools
import logging
import os
from tensorflow.keras import mixed_precision

from Hyperparameter import Hyperparameter
from PreprocessMultiChannelMILImageData import PreprocessMultiChannelMILImageData
from PreprocessMILImageData import PreprocessMILImageData
from mil_postprocessing import MilPostprocessor
from ImageModelBuilder import ImageModel
from PreprocessData import PreprocessData
from PreprocessRawData import PreprocessRawData
from RawModelBuilder import RawModel
import Callbacks


# hyperparameter_list = Hyperparameter(
#     downsample=["max_pool", "ave_pool"],  # , "stride"
#     activation=["relu_norm", "selu"],  # , "relu_norm", "relu_norm",
#     conv_layer=["lay4"],  # "lay4",
#     dropout=["little_drop"],  # "no_drop", , "lot_drop"
#     regularizer=["little_l2"],  # "l2",
#     reduction=["global_ave_pooling", "flatten"],  # "flatten", "global_ave_pooling", "ave_pooling_little",
#     first_layer=["n32"], # "n64"
#     init=["zeros"],  # , "same"
#     augment=["augment", "afalse"],  # "augment", "augment_strong",
#     noise=["noise"],  # "no_noise"
#     repetition=["fft_denoise3"],
#     residual=["rfalse", "residual"], # "residual",
#     mil=["mil"],
#     crop=["crop", "cfalse"],  #cfalse
#     normalize=["normalize"],  # nfalse
#     image_type=["structure", "combined", "images"],  #
#     label_smoothing=["label_smoothing", "lfalse"]
# )

#  TODO: Create Factory
hyperparameter_list = Hyperparameter(
    downsample=["stride"],  # , "stride"
    activation=["selu"],  # , "relu_norm", "relu_norm",
    conv_layer=["lay2"],  # "lay4",
    dropout=["no_drop"],  # "no_drop", , "lot_drop", little_drop
    regularizer=["little_l2"],  # "l2",
    reduction=["global_ave_pooling"],  # "flatten", "global_ave_pooling", "ave_pooling_little",
    first_layer=["n32"], # "n64"
    init=["zeros"],  # , "same"
    augment=["afalse"],  # "augment", "augment_strong",
    noise=["no_noise"],  # "no_noise"
    repetition=["fft_denoise3"],
    residual=["normal"], # "residual",
    mil=["mil"],
    crop=["cfalse"],  #cfalse
    normalize=["nfalse"],  # nfalse
    image_type=["raw"],  #
    label_smoothing=["label_smoothing"]
)


class ImageMain:

    def __init__(self):
        logging.basicConfig(filename='results/hyperparameter_study/hyperparameter_study.log', encoding='utf-8',
                            level=logging.INFO,
                            filemode="w")
        logging.info("BEGINNING HYPERPARAMTER STUDY")
        mixed_precision.set_global_policy('mixed_float16')
        self.model_folder = "results/hyperparameter_study/mil/models"

        # TODO:  look at slices of srtucture. Look at mean of raw data

        # TODO: these should all go into a class where only one model is calced
        self.image_size = None  # e.g. (204, 204, 1)
        self.model_name = None
        self.model_parameters = None
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self.image_type = None
        self.crop = None
        self.normalize = None
        self.class_weights = None
        self.augment = None

    # This is Hyperparameter Section
    def create_params_from_model_name(self):
        if "images" in self.model_name:
            self.image_type = "images"
        if "structure" in self.model_name:
            self.image_type = "structure"
        if "combined" in self.model_name:
            self.image_type = "combined"
        if "raw" in self.model_name:
            self.image_type = "raw"
        if "crop" in self.model_name:
            self.crop = 300
        else:
            self.crop = False
        if "normalize" in self.model_name:
            self.normalize = True
        else:
            self.normalize = False
        if "augment" in self.model_name:
            self.augment = True
        else:
            self.augment = False

    def generate_model_names(self):
        hyperparameter = list(hyperparameter_list.__dict__.values())
        all_hyper_combinations = list(itertools.product(*hyperparameter))
        random.shuffle(all_hyper_combinations)
        all_model_names = []
        all_model_parameters = []
        for i in all_hyper_combinations:
            all_model_names.append("_".join(i))
            all_model_parameters.append(i)
        return all_model_names, all_model_parameters

    def run_all(self):
        list_of_model_names, list_of_model_parameters = self.generate_model_names()
        for model_name, model_parameters in zip(list_of_model_names, list_of_model_parameters):
            logging.info(f"\n\nStarting calculation of {model_name}: \n\n")
            self.model_name = model_name
            self.model_parameters = model_parameters
            self.train()
            #self.eval()  # TODO

    def eval_all(self):
        list_of_model_names, list_of_model_parameters = self.generate_model_names()
        for model_name in list_of_model_names:
            self.model_name = model_name
            if os.path.exists(os.path.join(self.model_folder, model_name)):
                self.eval()

    # This is actual Main Section of Image Calculation
    def train(self):
        self.create_dataset("train")
        self.train_model()
        self.eval()

    def eval(self):
        self.create_dataset("test")
        #self.eval_dataset(self.ds_test, "test")
        #self.eval_dataset(self.ds_train.take(2), "train")
        #self.create_dataset("train")
        #self.ds_test = self.ds_train.take(2)
        self.eval_model()

    def create_dataset(self, data_type="train"):
        self.create_params_from_model_name()

        # TODO: create static method in rawdata class than factory
        #file_list = self.get_file_list(data_type)
        if data_type == "train":
            file_list, _ = PreprocessRawData.get_test_train_file_lists()
        else:
            _, file_list = PreprocessRawData.get_test_train_file_lists()

        if self.image_type == "combined":
            pid = PreprocessMultiChannelMILImageData(file_list, rgb=False, crop=self.crop, data_type=data_type,
                                         normalize=self.normalize, augment=self.augment)
        elif self.image_type == "raw":
            pid = PreprocessRawData(file_list, data_type=data_type)
        else:
            pid = PreprocessMILImageData(file_list, rgb=False, crop=self.crop, data_type=data_type,
                                         normalize=self.normalize, augment=self.augment)
        pid.preprocess_data_and_save()
        if data_type == "train":
            self.ds_train, self.ds_val = pid.create_dataset_for_calculation()
            self.class_weights = PreprocessRawData.calc_weights(pid.train_label_list)  # TODO factory
            #self.eval_dataset(self.ds_train)
            #self.eval_dataset(self.ds_val, "val")
        else:
            self.ds_test = pid.create_dataset_for_calculation()

    def get_file_list(self, data_type):
        if self.image_type == "combined":
            file_list_angio = PreprocessData.load_file_list(data_type, angio_or_structure="images")
            file_list_struc = PreprocessData.load_file_list(data_type, angio_or_structure="structure")
            file_list = PreprocessMultiChannelMILImageData.find_channel_pairs(file_list_angio, file_list_struc)
        else:
            file_list = PreprocessData.load_file_list(data_type, angio_or_structure=self.image_type)
        return file_list

    def train_model(self):
        #im = ImageModel(self.model_parameters)  # TODO factory
        #self.image_size = self.get_datasize()
        NUM_SAMPLES_TRAIN = 180
        NUM_SAMPLES_VAL = 15
        BATCH_SIZE = 1
        NUM_WALKTHROUGH = 3
        CHECKPOINTS = 30
        train_steps = int(NUM_SAMPLES_TRAIN * NUM_WALKTHROUGH / (BATCH_SIZE * CHECKPOINTS))
        val_steps = int(NUM_SAMPLES_VAL * NUM_WALKTHROUGH / (BATCH_SIZE * CHECKPOINTS))

        im = RawModel(self.model_parameters)  # TODO factory
        self.image_size = (204, 204, 1536, 1)
        model = im.model(output_to=logging.info, input_shape=self.image_size)
        model.fit(self.ds_train.batch(BATCH_SIZE).repeat(),
                  validation_data=self.ds_val.batch(BATCH_SIZE).repeat(),
                  epochs=50,
                  callbacks=Callbacks.raw_callback(self.model_name),  #TODO
                  class_weight=self.class_weights,
                  use_multiprocessing = True,
                  steps_per_epoch = train_steps,
                  validation_steps = val_steps
        )

    def eval_model(self):
        """
        Order of visualize_file_list needs to match order of test dataset!
        :return:
        """
        # TODO:
        visualize_file_list = PreprocessMILImageData.load_file_list("test", angio_or_structure="images")
        visualize_file_list = sorted(visualize_file_list, key=lambda file: (int(file[1]),
                                                                            int(file[0].split("_")[-1][:-4])))
        # TODO: sort file list and dataset together (should already be sorted) just pack into loop
        hp = MilPostprocessor(self.model_name, self.ds_test, visualize_file_list, crop=self.crop)
        hp.mil_raw_postprocessing()
        if "structure" in self.model_name or "combined" in self.model_name:
            visualize_file_list = PreprocessMILImageData.load_file_list("test", angio_or_structure="structure")
            visualize_file_list = sorted(visualize_file_list, key=lambda file: (int(file[1]),
                                                                                int(file[0].split("_")[-1][:-4])))
            hp = MilPostprocessor(self.model_name, self.ds_test, visualize_file_list, crop=self.crop)
            hp.mil_postprocessing()

    def get_datasize(self):
        for one_element, label in self.ds_train.take(1):
            return one_element.shape

    def eval_dataset(self, plot_dataset, bonus="train"):
        """
        Plots the first 50 of the dataset for validation purposes into "data/check_current..."
        :param plot_dataset:
        :param i:
        :return:
        """
        for k, data in enumerate(plot_dataset.take(50)):
            my_data = data[0].numpy().squeeze()
            if self.image_type == "combined":
                for i in range(2):
                    self.create_plot(my_data[..., i], data[1].numpy(), k + i*50, bonus)
            elif self.image_type == "raw":
                plt.figure(figsize=(12, 12))
                plt.grid(False)
                plt.plot(my_data[63, 135, :], cmap="gray")
                plt.savefig(f"data/check_current_dataset/processed{k}_{bonus}.png")
                plt.close()
            else:
                self.create_plot(my_data, data[1].numpy(), k, bonus)

    def create_plot(self, image, label, k, bonus):
        plt.figure(figsize=(12, 12))
        plt.grid(False)
        plt.imshow(image, cmap="gray")
        plt.title(f"{label}")
        plt.colorbar()
        plt.savefig(f"data/check_current_dataset/processed{k}_{bonus}.png")
        plt.close()


if __name__ == "__main__":
    os.chdir("../")
    keras.backend.clear_session()
    run_image = ImageMain()
    run_image.eval_all()
