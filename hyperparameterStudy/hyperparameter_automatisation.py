import itertools
import random
from Hyperparameter import Hyperparameter
from ImageModelBuilder import ImageModel
import logging
from image_dataset import ImageDataset
import Callbacks
from hyper_postprocessing import HyperPostprocessor
import os

hyperparameter_list = Hyperparameter(
    downsample=["ave_pool"],  # , "stride"
    activation=["selu"],  # , "relu_norm", "relu_norm",
    conv_layer=["lay6"],  # "lay4",
    dropout=["little_drop"],  # "no_drop", , "lot_drop"
    regularizer=["little_l2"],  # "l2",
    reduction=["global_ave_pooling"],  # "flatten", "global_ave_pooling", "ave_pooling_little",
    first_layer=["n64"], # "n64"
    init=["zeros"],  # , "same"
    augment=["augment"],  # "augment", "augment_strong",
    noise=["noise"],  # "no_noise"
    repetition=["second"]
)


def check_if_model_already_calced(name):
    calculate = True
    for root, dirs, files in os.walk("results/hyperparameter_study/postprocessing"):
        for sub_dirs in dirs:
            if name in sub_dirs:
                calculate = False
    return calculate


def hyperparameter_study(train=True, eval=False):
    logging.basicConfig(filename='results/hyperparameter_study/hyperparameter_study.log', encoding='utf-8',
                        level=logging.INFO,
                        filemode="w")
    logging.info("BEGINNING HYPERPARAMTER STUDY")
    my_dataset_class = ImageDataset()
    class_weights = my_dataset_class.calc_weights()
    hyperparameter = list(hyperparameter_list.__dict__.values())
    all_hyper_combinations = list(itertools.product(*hyperparameter))
    random.shuffle(all_hyper_combinations)

    for i in all_hyper_combinations:
        try:
            model_name = "_".join(i)
            logging.info(f"\n\nStarting calculation of {model_name}: \n\n")
            im = ImageModel(i)
            model = im.model(output_to=logging.info)
            dataset = ImageDataset.augment_from_param(my_dataset_class.dataset_train, i)
            if eval:
                eval_dataset(dataset, i)
            if check_if_model_already_calced(model_name) and train:
                train_model(model, dataset, my_dataset_class, model_name, class_weights)
        except BaseException as err:
            logging.info(f"{err}")
            logging.error(err)


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
    hp.all_models_postprocessing()


if __name__ == "__main__":
    hyperparameter_study(eval=True)
