import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as k
from Postprocessor import Postprocessing
import numpy as np
import skimage.transform as sk_tr
from base_postprocessing import BasePostprocessor
import mil_pooling
from PreprocessMILImageData import PreprocessMILImageData
from PreprocessData import PreprocessData
import copy

class MilPostprocessor(BasePostprocessor):

    def __init__(self, model_name, dataset, file_list, crop, threshold=None):
        super().__init__(model_name, "results/hyperparameter_study/mil", file_list)
        self.crop = crop
        self.test_ds = dataset  # Dataset needs to be sorted such that the first 100 elements are the elements of the first bag
        self.mil_pooling_model_name = "mil_pooling_" + model_name
        self.pooling_algo = None
        self.threshold = threshold

    def mil_postprocessing(self):
        #for model_ident in ["acc_", "loss_"]:
        model_ident = ""
        self.model = self.load_model(model_ident)

        prediction = self.create_prediction(self.model)
        prediction = prediction.reshape((-1, 10, 10))
        prediction = np.swapaxes(prediction, 1, 2).reshape((-1, 100))  # n predictions in the shape (n, 100)
        self.pooling_algo = mil_pooling.MilPooling(prediction, self.mil_pooling_model_name,
                                                   mil_pooling_type="max_pooling")  # TODO
        bag_prediction = self.pooling_algo.conduct_pooling() - self.threshold
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
            #self.plot_mil_auc(save_at)

    def plot_instances(self, path):
        new_path = os.path.join(path, "instances")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        prediction = self.create_prediction(self.model)
        for elem, prediction, i in zip(self.test_ds, prediction, range(len(self.test_file_list) * 100)):
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.imshow(elem[0], "gray")
            ax.axis(False)
            title_pred = np.round(prediction[0], 4)
            ax.set_title(f"{title_pred:.4f}")
            fig.savefig(os.path.join(new_path, os.path.basename(self.test_file_list[int(i/100)][0])[:-4] + f"_{i % 100}.png"), bbox_inches='tight')
            plt.close()

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
        self.plot_instance_predictions(instance_predictions, save_at)
        #self.plot_instances(save_at)
        self.calc_accuracy(instance_predictions, bag_predictions, save_at)
        self.plot_roc(save_at, bag_predictions)
        self.plot_mil_images(instance_predictions, bag_predictions, os.path.join(save_at, "grad_cam"))
        #self.plot_raw(instance_predictions, bag_predictions, os.path.join(save_at, "grad_cam"))
        self.plot_history(save_at)
        self.plot_mil_auc(save_at)
        self.plot_mil_real_distribution(instance_predictions, save_at)

    def plot_instance_predictions(self, instance_prediction, save_at):
        # 13 14 28 15 26 27
        import pandas as pd
        df = pd.DataFrame(instance_prediction)
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(20, 5))
        df = df.T
        tick_list, color_list = [], []
        no_diabetic = ["5577", "6338", "18832", "27719", "28065", "19077"]
        only_a_bit_diabetic = ["29122", "28832", "28810", "28477", "28133"]
        for path, label in self.test_file_list:
            if any([x in path for x in only_a_bit_diabetic]):
                label = "Minor"
                color_list.append("#FFAA60")
            elif any([x in path for x in no_diabetic]):
                label = "No"
                color_list.append("#9090FF")
            elif label == 1:
                label = "Major"
                color_list.append("#FF7070")
            elif label == 0:
                label = "Healthy"
                color_list.append("#70FF70")
            tick_list.append(label)
        ax = sns.boxplot(data=df, palette=color_list)
        ax.set_title("Instance Predictions Sorted by Bag:")
        ax.set(xlabel='DR Status', ylabel='Instance Predictions')
        ax.set_xticklabels(tick_list)
        plt.tight_layout()
        fig.savefig(os.path.join(save_at, "predictions.png"))


    def plot_mil_real_distribution(self, instance_predictions, save_at):
        pooling_algo = mil_pooling.MilPooling(instance_predictions, "mil_pooling_" + self.model_name,
                                                   mil_pooling_type="max_pooling")  # TODO
        bag_predictions = pooling_algo.conduct_pooling()
        self.plot_real_distribution(bag_predictions, save_at)

    def plot_mil_images(self, instance_predictions, bag_predictions, output_path):
        instance_predictions = instance_predictions.reshape((-1, 100))  # Reshape to (sample_no, 100)
        #max_min_prediction = (instance_predictions.max(), instance_predictions.min())
        postprocessor = Postprocessing(postprocessing_model=self.model)
        all_heatmap_plots, all_images = postprocessor.return_grad_cam(self.test_ds, visualize_layer=0)
        upper_bound = instance_predictions.max()
        crop = 0
        if self.crop:
            crop = 60 # 60
        image_size = 204  # 204
        for bag_prediction, prediction, bag_heatmaps, bag_images, file in sorted(zip(bag_predictions, instance_predictions,
                                                           np.swapaxes(all_heatmap_plots.reshape((-1, 10, 10, image_size-crop, image_size-crop)), 1, 2),
                                                           np.swapaxes(all_images.reshape((-1, 10, 10, image_size-crop, image_size-crop)), 1, 2),
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
                    #heat_im.set_clim(0, np.maximum(prediction.max(), 0.001))
            fig.suptitle(f"Image Type {file[1]} with Prediction: {bag_prediction}", fontsize=16)
            plt.subplots_adjust(wspace=0.02, hspace=0.02)
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
            heat_im = ax.imshow(sk_tr.resize(prediction, im.shape, order=0), "hot", alpha=0.4)
            heat_im.set_clim(0, up_lim)
            #heat_im.set_clim(0, np.maximum(prediction.max(), 0.001))
            ax.axis(False)
            fig.suptitle(f"Image Type {file[1]} with Prediction: {bag_prediction}", fontsize=16)
            plt.subplots_adjust(wspace=0, hspace=0)
            fig.savefig(os.path.join(os.path.join(output_path, ""), os.path.basename(file[0])), bbox_inches='tight')
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
        only_a_bit_diabetic = ["5577", "6338", "28133", "18832", "27719", "28065"]
        for predicted, truth in zip(instance_predictions.flatten(), self.test_ds):
            score += (predicted > 0 and truth[1].numpy() == 1.) or (predicted < 0 and truth[1].numpy() == 0.)
        instance_accuracy = score / instance_predictions.size
        score = 0
        for predicted, truth in zip(bag_predictions, self.test_file_list):
            if any([x in truth[0] for x in only_a_bit_diabetic]):
                score += predicted < 0
            else:
                score += (predicted > 0 and truth[1] == 1.) or (predicted < 0 and truth[1] == 0.)
        bag_accuracy = score / bag_predictions.size
        with open(os.path.join(output_to, "test_accuracy.txt"), "w") as f:
            f.write(f"Instance Accuracy: {instance_accuracy}\nBag Accuracy: {bag_accuracy}")
        return instance_accuracy, bag_accuracy

    def plot_mil_auc(self, output_path):
        sns.set_theme()
        history_df = pd.read_csv(self.history_path)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(history_df["epoch"], history_df["val_mil_metric"])
        ax.legend(["Validation Bag AUC"])
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, "mil_auc"))
        plt.close()

if __name__ == "__main__":
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
