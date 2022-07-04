import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as k
from hyperparameterStudy.image_dataset import ImageDataset
from Postprocessor import Postprocessing
import numpy as np
import Visualization
from hyperparameterStudy.base_postprocessing import BasePostprocessor


class HyperPostprocessor(BasePostprocessor):

    def __init__(self, model_name, dataset, file_list, results_folder="results/hyperparameter_study/supervised",
                 history_folder="results/hyperparameter_study"):
        super().__init__(model_name, results_folder, file_list, history_folder)
        self.test_ds = dataset
        #self.mil_pooling_model_name = "mil_pooling_" + model_name

    def supervised_processing(self):
        for model_ident in ["acc_", "loss_"]:
            model = self.load_model(model_ident)
            prediction = self.create_prediction(model)
            self.model_postprocessing(prediction, model, model_ident)

    def processing(self):
        model = self.load_model("")
        prediction = self.create_prediction(model)
        self.model_postprocessing(prediction, model, "", heat=False)

    def plot_heatmaps(self, model, output_path):
        sns.set_theme(style="white")
        mod_output_path = os.path.join(output_path, "grad_cam")
        postprocessor = Postprocessing(postprocessing_model=model)
        postprocessor.grad_cam_images(self.test_ds,
                                      Postprocessing.create_name_list_from_paths(self.test_file_list),
                                      folder_path=mod_output_path)

    def model_postprocessing(self, prediction, model, ident, qualify=True, heat=True):
        save_at = self.sort_prediction_into_folder(ident, qualify=qualify, prediction=prediction)
        self.plot_roc(save_at, prediction)
        # self.calc_accuracy(instance_predictions, bag_predictions, save_at)
        if heat:
            self.plot_heatmaps(model, save_at)
        self.plot_history(save_at)
        self.save_prediction(prediction, self.test_file_list, save_at)


if __name__ == "__main__":
    os.chdir("../")
    file_list = ImageDataset.load_file_list("test")
    dataset = ImageDataset(data_list=file_list, validation_split=False, mil=False)
    model_name = "max_pool_selu_lay6_little_drop_little_l2_global_ave_pooling_n32_same_no augment"
    hp = HyperPostprocessor(model_name, dataset, file_list)
    model = hp.load_model("acc_")
    acc_prediction = hp.create_prediction(model)
    hp.model_postprocessing(acc_prediction, model, "acc_", qualify=True)

# plot loss and accuracy

# Of all three models: best acc, best loss, last one
# make Prediction and see if all the outputs are zero -> Sort them out?

# Sort the models according to their accuracy?
