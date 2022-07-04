import itertools
import random
from Hyperparameter import Hyperparameter
from ImageModelBuilder import ImageModel
import logging
from image_dataset import ImageDataset
import Callbacks
from hyper_postprocessing import HyperPostprocessor
import os
import models2d
import tensorflow as tf
from mil_postprocessing import MilPostprocessor
from PreprocessImageData import PreprocessImageData
from PreprocessMILImageData import PreprocessMILImageData

hyperparameter_list = Hyperparameter(
    downsample=["ave_pool", "max_pool"],  # , "stride"
    activation=["selu", "relu_norm"],  # , "relu_norm", "relu_norm",
    conv_layer=["lay6"],  # "lay4",
    dropout=["no_drop"],  # "no_drop", , "lot_drop"
    regularizer=["little_l2"],  # "l2",
    reduction=["global_ave_pooling"],  # "flatten", "global_ave_pooling", "ave_pooling_little",
    first_layer=["n32"], # "n64"
    init=["zeros"],  # , "same"
    augment=["augment"],  # "augment", "augment_strong",
    noise=["noise"],  # "no_noise"
    repetition=["first"],
    residual=["residual"],
    mil=["mil"],
    crop=["crop", "cfalse"],
    normalize=["normalize", "nfalse"],
    image_type=["images", "structure"]
)

def check_if_model_already_calced(name):
    calculate = True
    for root, dirs, files in os.walk("results/hyperparameter_study/postprocessing"):
        for sub_dirs in dirs:
            if name in sub_dirs:
                calculate = False
    return calculate


def hyperparameter_study(train=True, eval=True, mobile_net=False):
    logging.basicConfig(filename='results/hyperparameter_study/hyperparameter_study.log', encoding='utf-8',
                        level=logging.INFO,
                        filemode="w")
    logging.info("BEGINNING HYPERPARAMTER STUDY")

    hyperparameter = list(hyperparameter_list.__dict__.values())
    all_hyper_combinations = list(itertools.product(*hyperparameter))
    random.shuffle(all_hyper_combinations)

    for i in all_hyper_combinations:
        #try:
        model_name = "_".join(i)
        logging.info(f"\n\nStarting calculation of {model_name}: \n\n")

        if "images" in model_name:
            image_type = "images"
        if "structure" in model_name:
            image_type = "structure"
        if "crop" in model_name:
            crop = 300
        else:
            crop = False
        if "normalize" in model_name:
            normalize = True
        else:
            normalize = False

        data_type = "train"
        file_list = ImageDataset.load_file_list(data_type, angio_or_structure=image_type)
        pid = PreprocessMILImageData(file_list, rgb=False, crop=crop, data_type=data_type, normalize=normalize)
        pid.preprocess_data_and_save()
        ds_train, ds_val = pid.create_dataset_for_calculation()
        class_weights = ImageDataset.calc_weights(ds_train)

        im = ImageModel(i)
        model = im.model(output_to=logging.info, input_shape=(204, 204, 1))
        if eval:
            eval_dataset(ds_val, i)
        if check_if_model_already_calced(model_name) and train:
            train_model(model, ds_train, ds_val, model_name, class_weights, image_type, crop, normalize)
        # except BaseException as err:
        #     logging.info(f"{err}")
        #     logging.error(err)


def converter_wrapper(image, label):
    image = skimage.color.gray2rgb(image)
    return image, label

import skimage
def resnet_test():
    my_dataset_class = ImageDataset()
    #class_weights = my_dataset_class.calc_weights()
    dataset = ImageDataset.augment_from_param(my_dataset_class.dataset_train, ["augment", "noise"])
    dataset = dataset.map(converter_wrapper)
    for elem in dataset.take(1):
        plt.imshow(elem[0].numpy()[:,:,0])


import matplotlib.pyplot as plt
def eval_dataset(plot_dataset, i):
    for k, b in enumerate(plot_dataset.take(50)):
        my_data = b[0].numpy().squeeze()
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        im1 = ax.imshow(my_data, cmap="gray")
        plt.title(f"{b[1].numpy()}")
        plt.colorbar(im1, ax=ax)
        fig.savefig(f"data/check_current_dataset/processed{k}.png")
        plt.close()


def train_model(model, ds_train, ds_val, model_name, class_weights, image_type, crop, normalize):
    model.fit(ds_train.batch(2),
              validation_data=ds_val.batch(1),
              epochs=50,
              callbacks=Callbacks.mil_pooling_callback(model_name),
              class_weight=class_weights)
    data_type = "test"
    file_list = sorted(ImageDataset.load_file_list(data_type, angio_or_structure=image_type))
    pid = PreprocessMILImageData(input_file_list=file_list, rgb=False, crop=crop, normalize=normalize,
                                 data_type=data_type)
    pid.preprocess_data_and_save()
    ds = pid.create_dataset_for_calculation()
    file_list = sorted(ImageDataset.load_file_list(data_type, angio_or_structure="images"))
    hp = MilPostprocessor(model_name, ds, file_list)
                          #sorted(file_list, key=lambda file: int(file[0].split("_")[-1].split(".")[0])))
    hp.mil_postprocessing()
    # data_type = "test"
    # file_list = ImageDataset.load_file_list(data_type)
    # pid = PreprocessImageData(file_list, mil=False, rgb=False, crop=False, data_type=data_type)
    # #pid.preprocess_data_and_save()
    # ds_test = pid.create_dataset_for_calculation()
    # hp = HyperPostprocessor(model_name, ds_test, file_list)
    # hp.supervised_processing()


def test_model():
    model_name = "ave_pool_relu_norm_lay4_no_drop_little_l2_flatten_n32_zeros_augment_noise_second_residual_mil"
    hp = HyperPostprocessor(model_name, mil=True)
    hp.mil_model_postprocessing()

def postprocess_all(search_in_folder, mil=False):
    my_model_names = os.listdir(search_in_folder)
    parsed_model_names = []
    for model_name in my_model_names:
        parsed_model_names.append(model_name.replace("acc_", "").replace("loss_", "").replace("mil_pooling_", ""))
    parsed_model_names = list(set(parsed_model_names))
    #file_list = ImageDataset.load_file_list("test")
    #ds_test = ImageDataset(data_list=file_list, validation_split=False, mil=mil)
    pid = PreprocessImageData(mil=False, rgb=False, crop=False, data_type=data_type)
    #pid.preprocess_data_and_save()
    ds_test = pid.create_dataset_for_calculation()
    for model_name in parsed_model_names:
        if mil:
            hp = MilPostprocessor(model_name, ds_test, pid.calculation_file_list)
            hp.mil_postprocessing()
        else:
            hp = HyperPostprocessor(model_name, ds_test, pid.calculation_file_list)
            hp.supervised_processing()



#def eval_model():

if __name__ == "__main__":
    os.chdir("../")
    #postprocess_all(search_in_folder="results/hyperparameter_study/best_model_image",
    #                mil=False)
    hyperparameter_study()
    import tensorflow.keras as k
    # data_type = "test"
    # file_list = ImageDataset.load_file_list(data_type)
    # pid = PreprocessMILImageData(rgb=False, crop=False, data_type=data_type)
    # ds = pid.create_dataset_for_calculation()
    # hp = MilPostprocessor("max_pool_selu_lay4_no_drop_little_l2_flatten_n32_zeros_augment_noise_third_residual_mil", ds,
    #                       sorted(file_list, key=lambda file: int(file[0].split("_")[-1].split(".")[0])))
    # hp.mil_postprocessing()


