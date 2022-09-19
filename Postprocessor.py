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
                                      display_labels=(0, 1)
                                      )
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, colorbar=False)
        plt.tight_layout()
        plt.savefig(name)

    # def non_binary_confusion(self, threshold=0, name="results/non_binary_confusion_matrix.png"):
    #     self.prediction_results = self.prediction_results > threshold
    #     self.prediction_results = self.belonging_labels
    #     self.prediction_results = [label if result for result, label in zip(self.prediction_results, self.belonging_labels)]
    #     cm = confusion_matrix(self.belonging_labels, self.prediction_results, labels=list(range(10)))
    #     print(cm)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                                   display_labels=list(range(10)))
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     disp.plot(ax=ax)
    #     plt.savefig(name)


    def return_grad_cam(self, dataset, name_list=None, visualize_layer=0, folder_path=None):
        layer_name_vis = self.get_conv_layer_name(visualize_layer)
        all_heatmap_plots = []
        all_images = []
        for image_label_pair in dataset:
            heatmap, class_out = self.grad_cam2(np.expand_dims(image_label_pair[0], axis=0), layer_name_vis)
            heatmap = np.maximum(heatmap, 0)
            rs_image = Postprocessing.min_max_scale(image_label_pair[0])[..., 0]
            rs_heatmap = Postprocessing.min_max_scale(heatmap)
            rs_heatmap = sk_tr.resize(rs_heatmap, rs_image.shape, order=0, mode="edge")
            all_heatmap_plots.append(rs_heatmap)
            all_images.append(rs_image)
        return np.array(all_heatmap_plots), np.array(all_images)

    def grad_cam_images(self, dataset, name_list=None, num_elements=50, visualize_layer=0, folder_path=None):
        layer_name_vis = self.get_conv_layer_name(visualize_layer)
        if name_list is None:
            name_list = range(500)
        for prediction, image_label_pair, ident in sorted(zip(self.prediction_results, dataset.take(num_elements), name_list)):
            heatmap, class_out = self.grad_cam(np.expand_dims(image_label_pair[0], axis=0), layer_name_vis)
            # heatmap, class_out = self.make_gradcam_heatmap(np.expand_dims(image_label_pair[0], axis=0))
            heatmap = np.maximum(heatmap, 0)
            rs_image = Postprocessing.min_max_scale(image_label_pair[0])[..., 0]
            rs_heatmap = Postprocessing.min_max_scale(heatmap) * np.abs(class_out.numpy().squeeze()) / \
                         np.maximum(self.prediction_results.max(), 0)
            rs_heatmap = sk_tr.resize(rs_heatmap, rs_image.shape)
            fig, ax = plt.subplots(1, 1, figsize=(18, 18))
            fig.suptitle(f"Image Type {image_label_pair[1]} with Prediction: {class_out[0][0]}", fontsize=16)
            # im1 = ax[0].imshow(rs_image, cmap="gray")
            # plt.colorbar(im1, ax=ax[0])
            # im2 = ax[1].imshow(heatmap, cmap="hot")
            # plt.colorbar(im2, ax=ax[1])
            ax.axis(False)
            ax.imshow(rs_image, cmap="gray")
            im3 = ax.imshow(rs_heatmap, cmap="hot", alpha=0.4)
            plt.subplots_adjust(wspace=0, hspace=0)
            #plt.colorbar(im3, ax=ax)
            # fig, ax = plt.subplots(1, 3, figsize=(40, 12))
            # fig.suptitle(f"Image Type {image_label_pair[1]} with Prediction: {class_out[0]}", fontsize=30)
            # im1 = ax[0].imshow(rs_image, cmap="gray")
            # plt.colorbar(im1, ax=ax[0])
            # im2 = ax[1].imshow(heatmap, cmap="hot")
            # plt.colorbar(im2, ax=ax[1])
            # ax[2].imshow(rs_image, cmap="gray")
            # im3 = ax[2].imshow(self.top_10_percent(rs_heatmap), cmap="hot", alpha=0.4)
            # plt.colorbar(im3, ax=ax[2])
            if folder_path is None:
                image_path = f"results/grad_cam/grad_cam{ident}.png"
            else:
                image_path = os.path.join(folder_path, f"grad_cam{ident}.png")
            plt.tight_layout()
            fig.savefig(image_path, bbox_inches='tight')
            plt.close()

    def only_grad_cam_overlay(self, image, visualize_layer=0):
        layer_name_vis = self.get_conv_layer_name(visualize_layer)
        heatmap, class_out = self.grad_cam(np.expand_dims(image, axis=0).astype("float32"), layer_name_vis)
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

    def grad_cam2(self, input_data, layer_name):
        """
        Calculate Gradient of Class out by last layer out. Average over all kernel. Greyscale.
        :param input_shape:
        :param input_data:
        :param layer_name:
        :return:
        """
        with tf.GradientTape() as tape:
            last_conv_layer_name = self.postprocessing_model.get_layer(layer_name)
            grad_cam_model = tf.keras.models.Model([self.postprocessing_model.inputs], [self.postprocessing_model.output
                , last_conv_layer_name.output])
            model_out, last_conv_layer = grad_cam_model(input_data)
            #tape.watch(last_conv_layer)
            #tape.watch(last_conv_layer)
            class_out = model_out[:, tf.argmax(model_out[0])]
            #tape.watch(model_out)
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = k.backend.mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = heatmap.numpy().reshape(last_conv_layer.shape[1:3].as_list())
        return heatmap, class_out

    def grad_cam(self, input_data, layer_name):
        """
        Calculate Gradient of Class out by last layer out. Average over all kernel. Greyscale.
        :param input_shape:
        :param input_data:
        :param layer_name:
        :return:
        """
        #self.postprocessing_model = k.applications.vgg16.VGG16()
        last_conv_layer = self.postprocessing_model.get_layer(layer_name)
        last_conv_layer_model = k.Model(inputs=self.postprocessing_model.input, outputs=last_conv_layer.output)

        # This is just weird I don't understand what I did here either (don't try to change it)
        #classifier_input = k.Input(shape=last_conv_layer.output.shape[1:])
        classifier_layer_names = []
        for layer in self.postprocessing_model.layers[::-1]:
            if layer.name == last_conv_layer.name:
                break
            classifier_layer_names.append(layer.name)

        classifier_input = k.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names[::-1]:
            x = self.postprocessing_model.get_layer(layer_name)(x)
        classifier_model = k.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(input_data)
            tape.watch(last_conv_layer_output)

            # 8. Get the class predictions and the class channel using the class index
            preds = classifier_model(last_conv_layer_output)
            grads = tape.gradient(preds, last_conv_layer_output)
            pooled_grads = k.backend.mean(grads, axis=(0, 1, 2))

            # 9. Using tape, Get the gradient for the predicted class wrt the output feature map of last conv layer
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
            heatmap = heatmap.numpy().reshape(last_conv_layer_output.shape[1:3].as_list())
        return heatmap, preds

    # def grad_cam(self, input_data, layer_name):
    #     """
    #     Calculate Gradient of Class out by last layer out. Average over all kernel. Greyscale.
    #     :param input_shape:
    #     :param input_data:
    #     :param layer_name:
    #     :return:
    #     """
    #     #self.postprocessing_model = k.applications.vgg16.VGG16()
    #     last_conv_layer = self.postprocessing_model.get_layer(layer_name)
    #     last_conv_layer_model = k.Model(inputs=self.postprocessing_model.input, outputs=last_conv_layer.output)
    #
    #     # This is just weird I don't understand what I did here either (don't try to change it)
    #     #classifier_input = k.Input(shape=last_conv_layer.output.shape[1:])
    #     classifier_layer_names = []
    #     for layer in self.postprocessing_model.layers[::-1]:
    #         if layer.name == last_conv_layer.name:
    #             break
    #         classifier_layer_names.append(layer.name)
    #
    #     classifier_input = k.Input(shape=last_conv_layer.output.shape[1:])
    #     x = classifier_input
    #     for layer_name in classifier_layer_names[::-1]:
    #         x = model.get_layer(layer_name)(x)
    #     classifier_model = k.Model(classifier_input, x)
    #
    #     with tf.GradientTape() as tape:
    #         last_conv_layer_output = last_conv_layer_model(input_data)
    #         tape.watch(last_conv_layer_output)
    #
    #         # 8. Get the class predictions and the class channel using the class index
    #         preds = classifier_model(last_conv_layer_output)
    #         class_channel = preds
    #
    #         # 9. Using tape, Get the gradient for the predicted class wrt the output feature map of last conv layer
    #     grads = tape.gradient(
    #         class_channel,
    #         last_conv_layer_output
    #     )
    #     pooled_grads = k.backend.mean(grads, axis=(0, 1, 2, 3))
    #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer.output), axis=-1)
    #     heatmap = heatmap.numpy().reshape(last_conv_layer.shape[1:3].as_list())
    #     return heatmap, class_channel

    def make_gradcam_heatmap(self,
            img_array
    ):
        # # 1. Create a model that maps the input image to the activations of the last convolution layer - Get last conv layer's output dimensions
        # last_conv_layer = self.postprocessing_model.get_layer("block2_conv2")
        # last_conv_layer_model = k.Model(model.inputs, last_conv_layer.output)
        #
        # # 2. Create another model, that maps from last convolution layer to the final class predictions - This is the classifier model that calculated the gradient
        # classifier_input = k.Input(shape=last_conv_layer.output.shape[1:])
        # classifier_layer_names = ["global_average_pooling2d", "dropout", "dense", "dense_1"]
        # x = classifier_input
        # for layer_name in classifier_layer_names:
        #     x = model.get_layer(layer_name)(x)
        # classifier_model = k.Model(classifier_input, x)

        # 1. Create a model that maps the input image to the activations of the last convolution layer - Get last conv layer's output dimensions
        last_conv_layer = self.postprocessing_model.get_layer("block2_conv2")
        last_conv_layer_model = k.Model(model.inputs, last_conv_layer.output)

        # 2. Create another model, that maps from last convolution layer to the final class predictions - This is the classifier model that calculated the gradient
        classifier_input = k.Input(shape=last_conv_layer.output.shape[1:])
        for layer in self.postprocessing_model.layers[::-1]:
            if layer.name == "block2_conv2":
                break
            print(layer.name)
        classifier_layer_names = ["global_average_pooling2d", "dropout", "dense", "dense_1"]
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = k.Model(last_conv_layer.output, model.outputs)

        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)

            # 8. Get the class predictions and the class channel using the class index
            preds = classifier_model(last_conv_layer_output)
            class_channel = preds

        # 9. Using tape, Get the gradient for the predicted class wrt the output feature map of last conv layer
        grads = tape.gradient(
            class_channel,
            last_conv_layer_output
        )

        # 10. Calculate the mean intensity of the gradient over its feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()

        # # 11. Multiply each channel in feature map array by weight importance of the channel
        # for i in range(pooled_grads.shape[-1]):
        #     last_conv_layer_output[:, :, i] *= pooled_grads[i]
        #
        # # 12. The channel-wise mean of the resulting feature map is our heatmap of class activation
        # heatmap = np.mean(last_conv_layer_output, axis=-1)
        #
        # # 13. Normalize the heatmap between [0, 1] for ease of visualization
        # heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        #
        # heatmaps.append({
        #     "class_id": class_indices[index],
        #     "heatmap": heatmap
        # })

        return heatmaps

    def get_conv_layer_name(self, conv_layer_number) -> str:
        i = 0
        for layer in self.postprocessing_model.layers[::-1]:
            if layer.__class__.__name__ == "Conv2D":
                if i == conv_layer_number:
                    return layer.name
                else:
                    i += 1
            if layer.__class__.__name__ == "Functional":
                for sub_layer in layer.layers[::-1]:
                    if sub_layer.__class__.__name__ == "Conv2D":
                        if i == conv_layer_number:
                            return sub_layer.name
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
            if image_identifier == "png":
                image_identifier = image_name.split(".")[-3].split("_")[-1]
            name_list.append(image_identifier)
        return name_list

if __name__ == "__main__":
    # with tf.device("cpu:0"):
    from vgg16 import create_vgg_model
    from PreprocessData import PreprocessData
    from PreprocessImageData import PreprocessImageData
    from PreprocessMILImageData import PreprocessMILImageData
    # model = create_vgg_model(input_shape=(2044, 2048, 3))
    # model.layers[1].summary()
    # model.summary()
    # for layer in model.layers:
    #     layer.trainable = True
    model = k.models.load_model("results/hyperparameter_study/mil/models/ave_pool_selu_lay4_no_drop_little_l2_global_ave_pooling_n32_zeros_augment_no_noise_fft_denoise7_residual_mil_cfalse_nfalse_images_lfalse")
    #model.summary()

    #tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
    file_list = PreprocessData.load_file_list(test_or_train="test", angio_or_structure="images")
    pid = PreprocessMILImageData(file_list, rgb=False, crop=False)
    pid._buffer_folder = "tests/test"
    pid.preprocess_data_and_save()
    ds = pid.create_dataset_for_calculation()
    # model.fit(ds.batch(1), epochs=2)

    # TODO I HAVE TO CALL tf.keras.applications.vgg16.preprocess_input

    #im, la = pid.preprocess_dataset(*file_list[0])
    # for elem in ds.take(1):
    #     plt.figure()
    #     plt.imshow(elem[0][:, :, 0], "gray")
    #     plt.figure()
    #     plt.hist(elem[0].numpy().flatten(), bins=50)
    # plt.show()

    pp = Postprocessing(prediction_results=np.zeros((len(file_list), )),
                        belonging_labels=np.zeros((len(file_list), )),
                        postprocessing_model=model)
    pp.return_grad_cam(dataset=ds, folder_path="tests/", visualize_layer=0)


