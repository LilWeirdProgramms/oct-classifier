import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as k
from image_dataset import ImageDataset
from Postprocessor import Postprocessing
import numpy as np
import Visualization


class BasePostprocessor:

    def __init__(self, model_name, path_root, file_list=None):
        self.model_name = model_name
        self.path_root = path_root
        self.history_path = os.path.join("results/hyperparameter_study", f"histories/{self.model_name}.csv")
        self.best_models_path = os.path.join(self.path_root, "models")
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
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        ax[0].plot(history_df["epoch"], history_df["loss"])
        ax[0].plot(history_df["epoch"], history_df["val_loss"])
        ax[0].legend(["Training Loss", "Validation Loss"])
        ax[1].plot(history_df["epoch"], history_df["accuracy"])
        ax[1].plot(history_df["epoch"], history_df["val_accuracy"])
        ax[1].legend(["Training Accuracy", "Validation Accuracy"])
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, "history"))
        plt.close()

    def create_prediction(self, model):
        out = model.predict(self.test_ds.dataset_train.batch(1), verbose=1)
        return out

    def load_model(self, model_type):
        full_loss_model_path = os.path.join(self.best_models_path, model_type + self.model_name)
        model = k.models.load_model(full_loss_model_path)
        return model

    # TODO: Implement for MIL
    def plot_heatmaps(self, model, output_path):
        sns.set_theme(style="white")
        mod_output_path = os.path.join(output_path, "grad_cam")
        postprocessor = Postprocessing(postprocessing_model=model)
        postprocessor.grad_cam_images(self.test_ds.dataset_train,
                                      Postprocessing.create_name_list_from_paths(self.test_file_list),
                                      folder_path=mod_output_path)

    def sort_prediction_into_folder(self, ident, qualify=False, prediction=None):
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
        bag_predictions = bag_predictions.reshape((bag_predictions.size, ))
        only_a_bit_diabetic = ["27060", "18749", "14590"]
        result_list = list(zip(bag_predictions, self.test_ds.data_list))
        very_diabetic = sorted([one_elem for one_elem in result_list if one_elem[1][1] == 1 and
                                not any(x in one_elem[1][0] for x in only_a_bit_diabetic)])
        very_diabetic_threshold = np.mean([number for number, _ in very_diabetic[:2]])
        same_images = ["15119", "14557"]
        same_image_score = []
        score = 0
        for predicted, truth in result_list:
            if any(x in truth[0] for x in only_a_bit_diabetic):
                score += 2 * (predicted < very_diabetic_threshold)
            else:
                score += (predicted > 0 and truth[1] == 1.) or (predicted < 0 and truth[1] == 0.)
            if any(x in truth[0] for x in same_images):
                same_image_score.append(predicted)
        score += 2 * (abs(same_image_score[0] - same_image_score[1]) < 0.2 * bag_predictions.std())
        evaluated_folder = str(score)
        return evaluated_folder

    def create_postprocessing_folder(self, classification, model_type):
        output_path = f"{self.path_root}/postprocessing/{classification}/{model_type}{self.model_name}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        grad_cam_path = os.path.join(output_path, "grad_cam")  # :(
        if not os.path.exists(grad_cam_path):
            os.makedirs(grad_cam_path)
        return output_path

    def save_prediction(self, prediction: np.ndarray, output_path):
        np.savetxt(os.path.join(output_path, "prediction.csv"), prediction)

    def model_postprocessing(self, prediction, model, ident, qualify=True):
        raise NotImplementedError("This is Abstract Class!!1")

    def all_models_postprocessing(self):
        acc_model = self.load_model("acc_")
        loss_model = self.load_model("loss_")
        acc_prediction = self.create_prediction(acc_model)
        loss_prediction = self.create_prediction(loss_model)
        self.model_postprocessing(acc_prediction, acc_model, "acc")
        self.model_postprocessing(loss_prediction, loss_model, "loss")
