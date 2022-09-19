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
        model = self.load_model("")
        self.prediction = self.create_prediction(model)

        #self.mil_pooling_model_name = "mil_pooling_" + model_name

    def supervised_processing(self):
        for model_ident in ["acc_", "loss_"]:
            model = self.load_model(model_ident)
            prediction = self.create_prediction(model)
            self.model_postprocessing(prediction, model, model_ident)

    def processing(self):
        self.model_postprocessing(self.prediction, model, "", heat=False)

    def processing_vgg(self):
        model = self.load_model("")
        for layer in model.layers:
            layer.trainable = True
        prediction = self.create_prediction(model)
        self.model_postprocessing(prediction, model, "", heat=True)

    def plot_heatmaps(self, model, output_path):
        sns.set_theme(style="white")
        mod_output_path = os.path.join(output_path, "grad_cam")
        postprocessor = Postprocessing(prediction_results=self.prediction, postprocessing_model=model)
        postprocessor.grad_cam_images(self.test_ds,
                                      Postprocessing.create_name_list_from_paths(self.test_file_list),
                                      folder_path=mod_output_path)

    def model_postprocessing(self, prediction, model, ident, qualify=True, heat=True):
        save_at = self.sort_prediction_into_folder(ident, qualify=qualify, prediction=prediction)
        self.plot_roc(save_at, prediction)
        self.calc_accuracy(prediction, save_at)
        #if heat:
        #    self.plot_heatmaps(model, save_at)
        self.plot_history(save_at)
        self.save_prediction(prediction, self.test_file_list, save_at)
        self.plot_real_distribution(prediction, save_at)

    def calc_accuracy(self, prediction, output_to):
        from sklearn import metrics

        tpr, fpr, threshold = metrics.roc_curve([label for path, label in self.test_file_list], prediction)
        gmean = np.sqrt(tpr * (1 - fpr))
        index = np.argmax(gmean)
        best_threshold = threshold[index]

        score = 0
        only_a_bit_diabetic = ["2666", "5577", "6338", "28133", "18832", "27719", "28065", "19077", "28477", "1252", "1082"]
        for predicted, truth in zip(prediction, self.test_file_list):
            if any([x in truth[0] for x in only_a_bit_diabetic]):
                score += predicted < best_threshold
            else:
                score += (predicted > best_threshold and truth[1] == 1.) or (predicted < best_threshold and truth[1] == 0.)
        bag_accuracy = score / prediction.size
        with open(os.path.join(output_to, "test_accuracy.txt"), "w") as f:
            f.write(f"Accuracy: {bag_accuracy}")
        return bag_accuracy



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
