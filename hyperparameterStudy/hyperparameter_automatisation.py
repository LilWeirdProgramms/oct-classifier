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

hyperparameter_list = Hyperparameter(
    downsample=["ave_pool", "max_pool"],  # , "stride"
    activation=["selu", "relu_norm", "relu"],  # , "relu_norm", "relu_norm",
    conv_layer=["lay4", "lay5"],  # "lay4",
    dropout=["no_drop"],  # "no_drop", , "lot_drop"
    regularizer=["little_l2"],  # "l2",
    reduction=["global_ave_pooling", "flatten"],  # "flatten", "global_ave_pooling", "ave_pooling_little",
    first_layer=["n32"], # "n64"
    init=["zeros"],  # , "same"
    augment=["augment"],  # "augment", "augment_strong",
    noise=["noise"],  # "no_noise"
    repetition=["second"],
    residual=["residual"],
    mil=["mil"]
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
    with tf.device("cpu:0"):
        my_dataset_class = ImageDataset(mil=True)
        class_weights = my_dataset_class.calc_weights()
    hyperparameter = list(hyperparameter_list.__dict__.values())
    all_hyper_combinations = list(itertools.product(*hyperparameter))
    random.shuffle(all_hyper_combinations)

    for i in all_hyper_combinations:
        try:
            model_name = "_".join(i)
            logging.info(f"\n\nStarting calculation of {model_name}: \n\n")
            if mobile_net:
                model = models2d.my_mobile_net_model((140, 140, 1), trainable=True)
            else:
                im = ImageModel(i)
                model = im.model(output_to=logging.info, input_shape=(140, 140, 1))
            dataset = ImageDataset.augment_from_param(my_dataset_class.dataset_train, i)
            if eval:
                eval_dataset(dataset, i)
            if check_if_model_already_calced(model_name) and train:
                train_model(model, dataset, my_dataset_class, model_name, class_weights)
        except BaseException as err:
            logging.info(f"{err}")
            logging.error(err)


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
    k = 0
    ds = plot_dataset.take(10)
    for b in ds:
        my_data = b[0].numpy().squeeze()
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        im1 = ax.imshow(my_data, cmap="gray", vmin=-2, vmax=3)
        plt.colorbar(im1, ax=ax)
        fig.savefig(f"data/check_current_dataset/processed{k}.png")
        plt.close()
        k += 1


def train_model(model, plot_dataset, my_dataset_class, model_name, class_weights):
    model.fit(plot_dataset.batch(2),
              validation_data=my_dataset_class.dataset_val.batch(1),
              epochs=100,
              callbacks=Callbacks.hyper_image_callback(model_name),
              class_weight=class_weights)
    hp = HyperPostprocessor(model_name)
    hp.mil_model_postprocessing()

def test_model():
    model_name = "ave_pool_relu_lay4_no_drop_little_l2_global_ave_pooling_n32_zeros_augment_noise_second_residual_mil"
    hp = HyperPostprocessor(model_name, mil=True)
    hp.mil_model_postprocessing()



if __name__ == "__main__":
    test_model()
