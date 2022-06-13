import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as k
from image_dataset import ImageDataset
from Postprocessor import Postprocessing
import numpy as np
import Visualization
from base_postprocessing import BasePostprocessor

class HyperPostprocessor(BasePostprocessor):

    def __init__(self, model_name):
        super().__init__(model_name, "results/hyperparameter_study/supervised")
        self.test_ds = ImageDataset(data_list=self.test_file_list, validation_split=False, mil=False)

    def supervised_processing(self):
        for model_ident in ["acc_", "loss_"]:
            model = self.load_model(model_ident)
            prediction = self.create_prediction(model)
            self.model_postprocessing(prediction, model, model_ident)

    def plot_heatmaps(self, model, output_path):
        sns.set_theme(style="white")
        mod_output_path = os.path.join(output_path, "grad_cam")
        postprocessor = Postprocessing(postprocessing_model=model)
        postprocessor.grad_cam_images(self.test_ds.dataset_train,
                                      Postprocessing.create_name_list_from_paths(self.test_file_list),
                                      folder_path=mod_output_path)

    def model_postprocessing(self, prediction, model, ident, qualify=True):
        save_at = self.sort_prediction_into_folder(ident, qualify=qualify, prediction=prediction)
        #self.calc_accuracy(instance_predictions, bag_predictions, save_at)
        self.plot_heatmaps(model, save_at)
        self.plot_history(save_at)


if __name__ == "__main__":
    os.chdir("../")
    model_name = "ave_pool_selu_lay6_little_drop_little_l2_ave_pooling_large_n32_zeros_augment_noise"
    hp = HyperPostprocessor(model_name)
    model = hp.load_model("acc_")
    acc_prediction = hp.create_prediction(model)
    hp.model_postprocessing(acc_prediction, model, "acc_", qualify=True)

# plot loss and accuracy

# Of all three models: best acc, best loss, last one
# make Prediction and see if all the outputs are zero -> Sort them out?

# Sort the models according to their accuracy?
