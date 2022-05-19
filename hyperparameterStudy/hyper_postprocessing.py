import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as k
from image_dataset import ImageDataset
from Postprocessor import Postprocessing
import numpy as np
import Visualization


class HyperPostprocessor:

    def __init__(self, name, mil=False):
        self.name = name
        path_root = "results/hyperparameter_study"
        self.history_path = os.path.join(path_root, "histories")
        self.specific_history = self.name + ".csv"
        self.best_models_path = os.path.join(path_root, "best_model_image")
        self.test_file_list = self.read_test_files()
        self.test_ds = ImageDataset(data_list=self.test_file_list, validation_split=False, mil=mil)

    def read_test_files(self):
        folder = "data/diabetic_images/test_files"
        files = sorted(os.listdir(folder))
        return [(os.path.join(folder, file), -1) for file in files]

    def plot_history(self, output_path):
        sns.set_theme()
        full_history_path = os.path.join(self.history_path, self.specific_history)
        history_df = pd.read_csv(full_history_path)
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
        full_loss_model_path = os.path.join(self.best_models_path, model_type + self.name)
        model = k.models.load_model(full_loss_model_path)
        return model

    def plot_heatmaps(self, model, output_path):
        sns.set_theme(style="white")
        mod_output_path = os.path.join(output_path, "grad_cam")
        postprocessor = Postprocessing(postprocessing_model=model)
        postprocessor.grad_cam_images(self.test_ds.dataset_train,
                                      Postprocessing.create_name_list_from_paths(self.test_file_list),
                                      folder_path=mod_output_path)

    def evaluate_prediction(self, tp):
        """

        :param tp:
        Predicts:
        Healthy links, Healthy rechts
        Diabetic 2x different contrast
        2x very diabetic
        :return:
        """
        healthy_file1, healthy_file2 = 0, 1
        very_diabetic_file1, very_diabetic_file2 = 4, 5
        score = 0
        score += int(tp[healthy_file1] < 0 and tp[healthy_file2] < 0)  # healthy == healthy
        score += int(tp[very_diabetic_file1] > 0 and tp[very_diabetic_file2] > 0)  # diabetic == diabetic
        score += int(tp[4] > tp[2] and tp[4] > tp[3])
        score += int(tp[5] > tp[2] and tp[5] > tp[3])
        score += int(tp[5] > tp[4])
        score += int(np.argmin(abs(tp - tp[2])) == 3)  # 2 und 3 close to each other
        score = 0 if round(np.sum(abs(tp - np.mean(tp))), 1) == 0.0 else score
        match score:
            case 0 | 1 | 2:
                evaluated_folder = "failed"
            case 3 | 4:
                evaluated_folder = "not_good"
            case 5 | 6:
                evaluated_folder = "possible_success"
        return evaluated_folder

    def create_postprocessing_folder(self, classification, model_type):
        output_path = f"results/hyperparameter_study/postprocessing/{classification}/{model_type}_{self.name}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        grad_cam_path = os.path.join(output_path, "grad_cam")
        if not os.path.exists(grad_cam_path):
            os.makedirs(grad_cam_path)
        return output_path

    def save_prediction(self, prediction: np.ndarray, output_path):
        np.savetxt(os.path.join(output_path, "prediction.csv"), prediction)

    def model_postprocessing(self, prediction, model, ident, qualify=True):
        if qualify:
            success_or_failed_folder = self.evaluate_prediction(prediction)
        else:
            success_or_failed_folder = "test_results"
        save_at = self.create_postprocessing_folder(success_or_failed_folder, ident)
        self.plot_heatmaps(model, save_at)
        self.plot_history(save_at)
        self.save_prediction(prediction, save_at)

    def all_models_postprocessing(self):
        acc_model = self.load_model("acc_")
        loss_model = self.load_model("loss_")
        acc_prediction = self.create_prediction(acc_model)
        print(acc_prediction)
        loss_prediction = self.create_prediction(loss_model)
        print(loss_prediction)
        self.model_postprocessing(acc_prediction, acc_model, "acc")
        self.model_postprocessing(loss_prediction, loss_model, "loss")

    def sort_image_by_prediction(self):
        self.mil_model_postprocessing()

    def mil_model_postprocessing(self):
        acc_model = self.load_model("acc_")
        #loss_model = self.load_model("loss_")
        acc_prediction = self.create_prediction(acc_model)
        acc_prediction = acc_prediction.reshape((-1, 100))
        output_path = f"results/mil/{'acc_' + self.name}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #self.plot_history(output_path)
        proto_file_list = [filename for filename, label in self.test_file_list]
        for file_name, prediction in zip(proto_file_list, acc_prediction):
            vis = Visualization.ImageVisualizer(results=prediction.reshape(10, 10).transpose().reshape(100, 1),
                                                image_size=(1400, 1400),
                                                instance_size=(140, 140, 10, 10),
                                                background_image_path=file_name,
                                                crop=(300 + 22, 300 + 24)
                                                )
            vis.plot_results_map(name=os.path.join(output_path, os.path.basename(file_name)))
            print("done")
        #print(acc_prediction)
        #loss_prediction = self.create_prediction(loss_model)

    def sort_predictions(self, output_path):
        i = 1
        for prediction, data in sorted(zip(acc_prediction, self.test_ds.train_data)):
            plt.figure(figsize=(12, 12))
            plt.title = f"{prediction}"
            plt.imsave(output_path + str(i) + ".png", np.squeeze(data))
            plt.close()
            i += 1



if __name__ == "__main__":
    os.chdir("../")
    model_name = "ave_pool_selu_lay6_little_drop_little_l2_ave_pooling_large_n32_zeros_augment_noise"
    hp = HyperPostprocessor(model_name)
    model = hp.load_model("acc_")
    acc_prediction = hp.create_prediction(model)
    hp.model_postprocessing(acc_prediction, model, "acc", qualify=False)

# plot loss and accuracy

# Of all three models: best acc, best loss, last one
# make Prediction and see if all the outputs are zero -> Sort them out?

# Sort the models according to their accuracy?
