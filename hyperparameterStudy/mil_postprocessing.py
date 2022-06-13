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
import mil_pooling
from mil_visualizer import MilVisualizer


class MilPostprocessor(BasePostprocessor):

    def __init__(self, model_name, dataset, file_list):
        super().__init__(model_name, "results/hyperparameter_study/mil", file_list)
        self.test_ds = dataset
        self.mil_pooling_model_name = "mil_pooling_" + model_name

    def mil_postprocessing(self):
        for model_ident in ["acc_", "loss_"]:
            model = self.load_model(model_ident)

            prediction = self.create_prediction(model)
            prediction = prediction.reshape((-1, 10, 10))
            prediction = np.swapaxes(prediction, 1, 2).reshape((-1, 100))

            pooling_algo = mil_pooling.MilPooling(prediction, self.mil_pooling_model_name)
            bag_prediction = pooling_algo.shallow_mil_pooling()
            self.model_postprocessing(prediction, bag_prediction, model_ident)

    def model_postprocessing(self, instance_predictions, bag_predictions, ident, qualify=True):
        """

        :param prediction:
        :param model:
        :param ident: One of acc or loss (What was the criteria for early stopping)
        :param qualify:
        :return:
        """
        save_at = self.sort_prediction_into_folder(ident, qualify=qualify, prediction=bag_predictions)
        self.calc_accuracy(instance_predictions, bag_predictions, save_at)
        self.plot_mil_images(instance_predictions, bag_predictions, os.path.join(save_at, "grad_cam"))
        self.plot_history(save_at)

    def plot_mil_images(self, instance_predictions, bag_predictions, output_path):
        instance_predictions = instance_predictions.reshape((-1, 100))  # Reshape to (sample_no, 100)
        max_min_prediction = (instance_predictions.max(), instance_predictions.min())
        for bag_prediction, prediction, file in sorted(zip(bag_predictions, instance_predictions, self.test_file_list),
                                                       key=lambda test_file: test_file[0]):
            attention_weights = self.get_attention_weights(prediction)
            mil_vis = MilVisualizer(background_image_tuple=file, instance_results=prediction.reshape((10, 10))
                                    , bag_result=bag_prediction
                                    , attention_map=attention_weights.reshape((10, 10))
                                    , max_min_prediction=max_min_prediction)
            mil_vis.create_figure(save_at=os.path.join(output_path, os.path.basename(file[0])))
            # vis = Visualization.ImageVisualizer(results=prediction.reshape(10, 10).transpose().reshape(100, 1),
            #                                     image_size=(1400, 1400),
            #                                     image_split=(10, 10),
            #                                     background_image_path=data_name,
            #                                     crop=(300 + 22, 300 + 24))
            # vis.plot_results_map(name=os.path.join(output_path, os.path.basename(data_name)),
            #                      title=f"Predicted: {bag_prediction}, True: {data_label}")

    def get_attention_weights(self, prediction):
        pooling_algo = mil_pooling.MilPooling(prediction, self.mil_pooling_model_name)
        attention_weights = pooling_algo.get_attention_weights(prediction.reshape((1, 100))).reshape((100, ))
        return attention_weights

    def calc_accuracy(self, instance_predictions, bag_predictions, output_to):
        score = 0
        for predicted, truth in zip(instance_predictions, self.test_ds.dataset_train):
            score += (predicted[0] > 0 and truth[1].numpy() == 1.) or (predicted[0] < 0 and truth[1].numpy() == 0.)
        instance_accuracy = score / instance_predictions.size
        score = 0
        for predicted, truth in zip(bag_predictions, self.test_ds.data_list):
            score += (predicted > 0 and truth[1] == 1.) or (predicted < 0 and truth[1] == 0.)
        bag_accuracy = score / bag_predictions.size
        with open(os.path.join(output_to, "test_accuracy.txt"), "w") as f:
            f.write(f"Instance Accuracy: {instance_accuracy}\nBag Accuracy: {bag_accuracy}")
        return instance_accuracy, bag_accuracy

if __name__ == "__main__":
    os.chdir("../")
    model_name = "ave_pool_relu_lay4_no_drop_little_l2_global_ave_pooling_n32_zeros_augment_noise_second_residual_mil"
    file_list = ImageDataset.load_file_list("test")
    mil_dataset = ImageDataset(data_list=file_list, validation_split=False, mil=True)
    hp = MilPostprocessor(model_name, mil_dataset, file_list)
    hp.mil_postprocessing()

# Instane and Bag Accuracy are criteria. Sivere vs not sivere cases is criteria.
