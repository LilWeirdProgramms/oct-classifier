import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import tensorflow.keras as k
import os
import numpy as np
import skimage.transform as sk_tr
import matplotlib


class Postprocessing:

    def __init__(self, prediction_results=None, belonging_labels=None, postprocessing_model: k.Model = None):
        self.prediction_results = prediction_results
        self.belonging_labels = belonging_labels
        self.postprocessing_model = postprocessing_model

    def binary_confusion_matrix(self, threshold=0, name="results/binary_confusion_matrix.png"):
        """

        :param threshold: Threshold at what probability an instance is decided as being diabetic
        :return:
        """
        self.prediction_results = self.prediction_results > threshold
        cm = confusion_matrix(self.belonging_labels, self.prediction_results, labels=(0, 1))
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax)
        plt.savefig(name)

    def grad_cam_images(self, dataset, name_list=None, num_elements=50, visualize_layer=0, folder_path=None):
        layer_name_vis = self.get_conv_layer_name(visualize_layer)
        if name_list is None:
            name_list = range(500)
        for image_label_pair, ident in zip(dataset.take(num_elements), name_list):
            heatmap, class_out = self.grad_cam(np.expand_dims(image_label_pair[0], axis=0), layer_name_vis)
            heatmap = np.maximum(heatmap, 0)
            rs_image = np.squeeze(Postprocessing.min_max_scale(image_label_pair[0]), axis=2)
            rs_heatmap = Postprocessing.min_max_scale(heatmap)
            rs_heatmap = sk_tr.resize(rs_heatmap, rs_image.shape)
            fig, ax = plt.subplots(1, 3, figsize=(40, 12))
            fig.suptitle(f"Image Type {image_label_pair[1]} with Prediction: {class_out[0]}", fontsize=30)
            im1 = ax[0].imshow(image_label_pair[0], cmap="gray")
            plt.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(heatmap, cmap="hot")
            plt.colorbar(im2, ax=ax[1])
            ax[2].imshow(rs_image, cmap="gray")
            im3 = ax[2].imshow(self.top_10_percent(rs_heatmap), cmap="hot", alpha=0.4)
            plt.colorbar(im3, ax=ax[2])
            if folder_path is None:
                image_path = f"results/grad_cam/grad_cam{ident}.png"
            else:
                image_path = os.path.join(folder_path, f"grad_cam{ident}.png")
            plt.tight_layout()
            fig.savefig(image_path)
            plt.close()

    def only_grad_cam_overlay(self, image, visualize_layer=0):
        layer_name_vis = self.get_conv_layer_name(visualize_layer)
        heatmap, class_out = self.grad_cam(np.expand_dims(image, axis=0), layer_name_vis)
        heatmap = np.maximum(heatmap, 0)
        rs_image = np.squeeze(Postprocessing.min_max_scale(image), axis=2)
        rs_heatmap = Postprocessing.min_max_scale(heatmap)
        rs_heatmap = sk_tr.resize(rs_heatmap, rs_image.shape, mode="edge", order=3)
        return rs_image, rs_heatmap

    def top_10_percent(self, heatmap: np.ndarray):
        threshold = 0.32 * np.max(heatmap)
        heatmap[heatmap < threshold] = 0
        return heatmap

    @staticmethod
    def min_max_scale(data): return (data - np.min(data)) / (np.max(data) - np.min(data))

    def grad_cam(self, input_data, layer_name):
        """
        Calculate Gradient of Class out by last layer out. Average over all kernel. Greyscale.
        :param input_shape:
        :param input_data:
        :param layer_name:
        :return:
        """
        with tf.GradientTape() as tape:
            last_conv_layer = self.postprocessing_model.get_layer(layer_name)
            grad_cam_model = tf.keras.models.Model([self.postprocessing_model.inputs], [self.postprocessing_model.output, last_conv_layer.output])
            model_out, last_conv_layer = grad_cam_model(input_data)
            class_out = model_out[:, tf.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = k.backend.mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = heatmap.numpy().reshape(last_conv_layer.shape[1:3].as_list())
        return heatmap, class_out

    def get_conv_layer_name(self, conv_layer_number) -> str:
        i = 0
        for layer in self.postprocessing_model.layers[::-1]:
            if layer.__class__.__name__ == "Conv2D":
                if i == conv_layer_number:
                    return layer.name
                else:
                    i += 1


    @staticmethod
    def create_name_list_from_paths(path_list):
        """
        Assumes '_ident.png' format
        :param path_list: List of Paths to the images used in postprocessing. Assumes an identifier number at the end
        :return: name_list: a list of identifiers that can be used to identify postprocessing images
        """
        name_list = []
        for path, label in path_list:
            image_name = os.path.basename(path)
            image_identifier = image_name.split(".")[-2].split("_")[-1]
            name_list.append(image_identifier)
        return name_list
