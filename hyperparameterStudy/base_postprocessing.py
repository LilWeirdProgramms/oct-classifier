import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as k
from Postprocessor import Postprocessing
import numpy as np
import Visualization
from sklearn import metrics
import models
from ImageModelBuilder import MilLoss

class BasePostprocessor:

    def __init__(self, model_name, path_root, file_list=None, history_path="results/hyperparameter_study"):
        self.model_name = model_name
        self.path_root = path_root
        self.history_path = os.path.join(history_path, f"histories/{self.model_name}.csv")
        self.best_models_path = os.path.join(self.path_root, "models")
        #self.best_models_path = "results/hyperparameter_study/best_model_image"
        if file_list is None:
            self.test_file_list = self.read_test_files()
        else:
            self.test_file_list = file_list
        self.test_ds = None

    def read_test_files(self):
        input_file_list = BasePostprocessor.input_list_from_folder("data/diabetic_images/test_files", 1) \
                          + BasePostprocessor.input_list_from_folder("data/healthy_images/test_files", 0)
        return input_file_list

    @staticmethod
    def input_list_from_folder(folder, label):
        files = os.listdir(folder)
        input_file_list = [(os.path.join(folder, file), label) for file in files]
        return input_file_list

    def plot_history(self, output_path):
        sns.set_theme()
        history_df = pd.read_csv(self.history_path)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history_df["epoch"], history_df["loss"])
        ax[0].plot(history_df["epoch"], history_df["val_loss"])
        ax[0].legend(["Training Loss", "Validation Loss"])
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[1].plot(history_df["epoch"], history_df["accuracy"])
        ax[1].plot(history_df["epoch"], history_df["val_accuracy"])
        ax[1].legend(["Training Accuracy", "Validation Accuracy"])
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        #  TODO: Tight Layout somehow throws error in debug mode (python 3.10 bug) (downgrade to solve)
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, "history"))
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(history_df["epoch"], history_df["precision"])
        ax.plot(history_df["epoch"], history_df["val_precision"])
        ax.legend(["Training Precision", "Validation Precision"])
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, "precision"))
        plt.close()


    def plot_roc(self, output_path, prediction):
        sns.set_theme()
        fig = plt.figure(figsize=(5, 5))
        labels = [label for path , label in self.test_file_list]
        fpr, tpr, threshold = metrics.roc_curve(labels, prediction, pos_label=1)
        auc = metrics.roc_auc_score(labels, prediction)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver operating characteristic with AUC = {round(auc, 5)}")
        fig.savefig(os.path.join(output_path, "roc_curve"))
        plt.close()
        fig.savefig(os.path.join(output_path, "history"))
        plt.close()

    def create_prediction(self, model):
        out = model.predict(self.test_ds.batch(1), verbose=1)
        return out

    def load_model(self, model_type):
        full_loss_model_path = os.path.join(self.best_models_path, model_type + self.model_name)
        model = k.models.load_model(full_loss_model_path, custom_objects={'MilMetric':models.MilMetric, "MilLoss":MilLoss},
                                    compile=False)
        return model

    # TODO: Implement for MIL
    def plot_heatmaps(self, model, output_path):
        sns.set_theme(style="white")
        mod_output_path = os.path.join(output_path, "grad_cam")
        postprocessor = Postprocessing(postprocessing_model=model)
        postprocessor.grad_cam_images(self.test_ds,
                                      Postprocessing.create_name_list_from_paths(self.test_file_list),
                                      folder_path=mod_output_path)

    def sort_prediction_into_folder(self, ident, qualify=True, prediction=None):
        if qualify:
            success_or_failed_folder = self.evaluate_prediction(prediction)
        else:
            success_or_failed_folder = "test_results"
        save_at = self.create_postprocessing_folder(success_or_failed_folder, ident)
        return save_at

    def evaluate_prediction(self, bag_predictions):
        """

        :param tp:
        Predicts:
        Healthy links, Healthy rechts
        Diabetic 2x different contrast
        2x very diabetic
        :return:
        """
        # 13 14 28 15 26 27
        bag_predictions = bag_predictions.reshape((bag_predictions.size, ))
        only_a_bit_diabetic = ["2666", "5577", "6338", "28133", "18832", "27719", "28065", "1252", "1082"]
        result_list = list(zip(bag_predictions, self.test_file_list))
        very_diabetic = sorted([one_elem for one_elem in result_list if one_elem[1][1] == 1 and
                                not any(x in one_elem[1][0] for x in only_a_bit_diabetic)])
        very_diabetic_threshold = np.mean([number for number, _ in very_diabetic[:2]])
        same_images = ["28477", "28810"]
        same_image_score = []
        score = 0
        for predicted, truth in result_list:
            if any([x in truth[0] for x in only_a_bit_diabetic]):
                score += 2 * (predicted < very_diabetic_threshold)
            else:
                score += (predicted > 0 and truth[1] == 1.) or (predicted < 0 and truth[1] == 0.)
            if any(x in truth[0] for x in same_images):
                same_image_score.append(predicted)
        score += 2 * (abs(same_image_score[0] - same_image_score[1]) < 0.2 * bag_predictions.std())
        evaluated_folder = str(score)
        return evaluated_folder

    def plot_real_distribution(self, prediction, output_path):
        no_diabetic = ["2666", "5577", "6338", "28133", "18832", "27719", "28065", "19077", "28477", "1252", "1082"]
        only_a_bit_diabetic = ["29122", "28832", "28810", "187", "32730"]
        real_class = []
        for truth in self.test_file_list:
            if any([x in truth[0] for x in no_diabetic]):
                real_class.append("No Signs")
            elif any([x in truth[0] for x in only_a_bit_diabetic]):
                real_class.append("Minor")
            # else:
            #     real_class.append(int(3 * truth[1]))
            elif truth[1] == 1:
                real_class.append("Severe")
            elif truth[1] == 0:
                real_class.append("Healthy")
        df = pd.DataFrame(
        {"Prediction": prediction.reshape(-1, ), "Underlying Class": real_class})
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(5, 5))
        sns.boxplot(x="Underlying Class", y="Prediction", data=df)
        plt.title("Predictions Sorted by Underlying Label")
        plt.xlabel("Underlying Class")
        plt.ylabel("Prediction")
        fig.savefig(os.path.join(output_path, "underlying.png"), bbox_inches='tight')
        plt.close()

    def create_postprocessing_folder(self, classification, model_type):
        output_path = f"{self.path_root}/postprocessing/{classification}/{model_type}{self.model_name}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        grad_cam_path = os.path.join(output_path, "grad_cam")  # :(
        if not os.path.exists(grad_cam_path):
            os.makedirs(grad_cam_path)
        return output_path

    def save_prediction(self, prediction: np.ndarray, file_list, output_path):
        save_lines = [f"{one_prediction}\t{ground_truth}\n" for
         one_prediction, ground_truth in zip(prediction.flatten(), file_list)]
        with open(os.path.join(output_path, "prediction.csv"), "w") as f:
            f.writelines(save_lines)
        #np.savetxt(os.path.join(output_path, "prediction.csv"), save_lines)

    def model_postprocessing(self, prediction, model, ident, qualify=True):
        raise NotImplementedError("This is Abstract Class!!1")

    def all_models_postprocessing(self):
        acc_model = self.load_model("acc_")
        loss_model = self.load_model("loss_")
        acc_prediction = self.create_prediction(acc_model)
        loss_prediction = self.create_prediction(loss_model)
        self.model_postprocessing(acc_prediction, acc_model, "acc")
        self.model_postprocessing(loss_prediction, loss_model, "loss")
