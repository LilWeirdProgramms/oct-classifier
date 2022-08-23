import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as k
from Postprocessor import Postprocessing
import numpy as np
import skimage.transform as sk_tr
import Visualization
from base_postprocessing import BasePostprocessor
import mil_pooling
from mil_visualizer import MilVisualizer
from PreprocessMILImageData import PreprocessMILImageData
from PreprocessData import PreprocessData
from PreprocessMultiChannelMILImageData import PreprocessMultiChannelMILImageData
import copy

class MilPostprocessor(BasePostprocessor):

    def __init__(self, model_name, dataset, file_list, crop):
        super().__init__(model_name, "results/hyperparameter_study/mil", file_list)
        # TODO: confirm file list has the same order as test_ds
        self.crop = crop
        self.test_ds = dataset  # Dataset needs to be sorted such that the first 100 elements are the elements of the first bag
        self.mil_pooling_model_name = "mil_pooling_" + model_name
        self.pooling_algo = None

    def mil_postprocessing(self):
        #for model_ident in ["acc_", "loss_"]:
        model_ident = ""
        self.model = self.load_model(model_ident)

        prediction = self.create_prediction(self.model)
        prediction = prediction.reshape((-1, 10, 10))
        prediction = np.swapaxes(prediction, 1, 2).reshape((-1, 100))  # n predictions in the shape (n, 100)

        self.pooling_algo = mil_pooling.MilPooling(prediction, self.mil_pooling_model_name,
                                                   mil_pooling_type="weighted")  # TODO
        bag_prediction = self.pooling_algo.conduct_pooling()
        self.model_postprocessing(prediction, bag_prediction, model_ident)

    def mil_raw_postprocessing(self):
        model_idents = ["prec_", "val_"]
        for model_ident in model_idents:
            self.model = self.load_model(model_ident)
            prediction = self.create_prediction(self.model)
            prediction = prediction.reshape((-1, 10, 10))
            self.pooling_algo = mil_pooling.MilPooling(prediction, self.mil_pooling_model_name,
                                                       mil_pooling_type="weighted")  # TODO
            bag_prediction = self.pooling_algo.conduct_pooling()
            save_at = self.sort_prediction_into_folder(model_ident, qualify=True, prediction=bag_prediction)
            self.calc_accuracy(prediction, bag_prediction, save_at)
            self.plot_roc(save_at, bag_prediction)
            self.plot_raw(prediction, bag_prediction, os.path.join(save_at, "grad_cam"))
            self.plot_history(save_at)


    def model_postprocessing(self, instance_predictions, bag_predictions, ident, qualify=True):
        """
prec_max_pool_relu_norm_lay4_little_drop_l2_global_ave_pooling_n32_zeros_afalse_no_noise_all4_4_residual_mil_cfalse_nfalse_raw_label_smoothing
prec_max_pool_relu_norm_lay4_little_drop_non_global_ave_pooling_n32_zeros_afalse_no_noise_all4_4_residual_mil_cfalse_nfalse_raw_label_smoothing
prec_max_pool_relu_norm_lay4_little_drop_l2_global_ave_pooling_n32_zeros_afalse_no_noise_all4_4_residual_mil_cfalse_nfalse_raw_label_smoothing
        :param prediction:
        :param model:
        :param ident: One of acc or loss (What was the criteria for early stopping)
        :param qualify:
        :return:
        """
        save_at = self.sort_prediction_into_folder(ident, qualify=qualify, prediction=bag_predictions)
        self.calc_accuracy(instance_predictions, bag_predictions, save_at)
        self.plot_roc(save_at, bag_predictions)
        self.plot_mil_images(instance_predictions, bag_predictions, os.path.join(save_at, "grad_cam"))
        self.plot_history(save_at)

    def plot_mil_images(self, instance_predictions, bag_predictions, output_path):
        instance_predictions = instance_predictions.reshape((-1, 100))  # Reshape to (sample_no, 100)
        #max_min_prediction = (instance_predictions.max(), instance_predictions.min())
        postprocessor = Postprocessing(postprocessing_model=self.model)
        all_heatmap_plots, all_images = postprocessor.return_grad_cam(self.test_ds, visualize_layer=0)
        upper_bound = instance_predictions.max()
        crop = 0
        if self.crop:
            crop = 60
        for bag_prediction, prediction, bag_heatmaps, bag_images, file in sorted(zip(bag_predictions, instance_predictions,
                                                           np.swapaxes(all_heatmap_plots.reshape((-1, 10, 10, 204-crop, 204-crop)), 1, 2),
                                                           np.swapaxes(all_images.reshape((-1, 10, 10, 204-crop, 204-crop)), 1, 2),
                                                           self.test_file_list),
                                                       key=lambda test_file: test_file[0]):
            #attention_weights = self.get_attention_weights(prediction)
            prediction = prediction.reshape((10, 10))
            fig, ax = plt.subplots(10, 10, figsize=(18, 18))
            for i in range(10):
                for j in range(10):
                    ax[i, j].imshow(bag_images[i, j], cmap="gray")
                    scaled_heatmap = bag_heatmaps[i, j] * np.maximum(prediction[i, j], 0) / bag_heatmaps[i, j].max()
                    heat_im = ax[i, j].imshow(scaled_heatmap, cmap="hot", alpha=0.4)
                    ax[i, j].axis(False)
                    heat_im.set_clim(0, upper_bound)
            fig.suptitle(f"Image Type {file[1]} with Prediction: {bag_prediction}", fontsize=16)
            plt.subplots_adjust(wspace=0, hspace=0)
            fig.savefig(os.path.join(output_path, os.path.basename(file[0])), bbox_inches='tight')
            plt.close()
            # mil_vis = MilVisualizer(background_image_tuple=file, instance_results=prediction
            #                         , bag_result=bag_prediction
            #                         #, attention_map=attention_weights.reshape((10, 10))
            #                         , attention_map=None
            #                         , max_min_prediction=max_min_prediction
            #                         , crop=self.crop)
            # mil_vis.create_figure(save_at=os.path.join(output_path, os.path.basename(file[0])))
            # vis = Visualization.ImageVisualizer(results=prediction.reshape(10, 10).transpose().reshape(100, 1),
            #                                     image_size=(1400, 1400),
            #                                     image_split=(10, 10),
            #                                     background_image_path=data_name,
            #                                     crop=(300 + 22, 300 + 24))
            # vis.plot_results_map(name=os.path.join(output_path, os.path.basename(data_name)),
            #                      title=f"Predicted: {bag_prediction}, True: {data_label}")

    def plot_raw(self, instance_predictions, bag_predictions, output_path):
        up_lim = instance_predictions.max()
        instance_predictions = instance_predictions.reshape((-1, 100))  # Reshape to (sample_no, 100)
        for bag_prediction, prediction, file in sorted(zip(bag_predictions, instance_predictions,
                                                           self.test_file_list),
                                                       key=lambda test_file: test_file[0]):
            prediction = prediction.reshape((10, 10))
            fig, ax = plt.subplots(1, 1, figsize=(18, 18))
            im = plt.imread(file[0])
            ax.imshow(im, "gray")
            heat_im = ax.imshow(sk_tr.resize(prediction, im.shape), "hot", alpha=0.4)
            heat_im.set_clim(0, up_lim)
            ax.axis(False)
            fig.suptitle(f"Image Type {file[1]} with Prediction: {bag_prediction}", fontsize=16)
            plt.subplots_adjust(wspace=0, hspace=0)
            fig.savefig(os.path.join(output_path, os.path.basename(file[0])), bbox_inches='tight')
            plt.close()
            # mil_vis = MilVisualizer(background_image_tuple=file, instance_results=prediction
            #                         , bag_result=bag_prediction
            #                         #, attention_map=attention_weights.reshape((10, 10))
            #                         , attention_map=None
            #                         , max_min_prediction=max_min_prediction
            #                         , crop=self.crop)
            # mil_vis.create_figure(save_at=os.path.join(output_path, os.path.basename(file[0])))
            # vis = Visualization.ImageVisualizer(results=prediction.reshape(10, 10).transpose().reshape(100, 1),
            #                                     image_size=(1400, 1400),
            #                                     image_split=(10, 10),
            #                                     background_image_path=data_name,
            #                                     crop=(300 + 22, 300 + 24))
            # vis.plot_results_map(name=os.path.join(output_path, os.path.basename(data_name)),
            #                      title=f"Predicted: {bag_prediction}, True: {data_label}")

    def get_attention_weights(self, prediction):
        attention_weights = self.pooling_algo.get_attention_weights(prediction.reshape((1, 100))).reshape((100, ))
        return attention_weights

    def calc_accuracy(self, instance_predictions, bag_predictions, output_to):
        score = 0
        for predicted, truth in zip(instance_predictions.flatten(), self.test_ds):
            score += (predicted > 0 and truth[1].numpy() == 1.) or (predicted < 0 and truth[1].numpy() == 0.)
        instance_accuracy = score / instance_predictions.size
        score = 0
        for predicted, truth in zip(bag_predictions, self.test_file_list):
            score += (predicted > 0 and truth[1] == 1.) or (predicted < 0 and truth[1] == 0.)
        bag_accuracy = score / bag_predictions.size
        with open(os.path.join(output_to, "test_accuracy.txt"), "w") as f:
            f.write(f"Instance Accuracy: {instance_accuracy}\nBag Accuracy: {bag_accuracy}")
        return instance_accuracy, bag_accuracy

if __name__ == "__main__":
    os.chdir("../")
    data_type = "test"
    model_name = "ave_pool_relu_norm_lay4_little_drop_little_l2_global_ave_pooling_n32_zeros_afalse_noise_fft_denoise3_rfalse_mil_cfalse_normalize_images_lfalse"
    file_list = PreprocessData.load_file_list(data_type, angio_or_structure="images")
    pid = PreprocessMILImageData(input_file_list=file_list, rgb=False, crop=False, data_type=data_type)
    pid.preprocess_data_and_save()
    ds = pid.create_dataset_for_calculation()


    visualize_file_list = PreprocessMILImageData.load_file_list("test", angio_or_structure="images")
    visualize_file_list = sorted(visualize_file_list, key=lambda file: (int(file[1]),
                                                                        int(file[0].split("_")[-1][:-4])))
    hp = MilPostprocessor(model_name, ds, visualize_file_list, crop=False)
    hp.mil_postprocessing()
    if "structure" in model_name or "combined" in model_name:
        visualize_file_list = PreprocessMILImageData.load_file_list("test", angio_or_structure="structure")
        visualize_file_list = sorted(visualize_file_list, key=lambda file: (int(file[1]),
                                                                            int(file[0].split("_")[-1][:-4])))
        hp = MilPostprocessor(model_name, ds, visualize_file_list, crop=300)
        hp.mil_postprocessing()

# Instane and Bag Accuracy are criteria. Sivere vs not sivere cases is criteria.
