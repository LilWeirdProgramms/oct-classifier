from hyperparameterStudy.image_dataset import ImageDataset
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


class MilVisualizer:

    def __init__(self, background_image_tuple: tuple,
                 instance_results: np.ndarray,
                 bag_result: np.ndarray,
                 attention_map: np.ndarray,
                 max_min_prediction: tuple
                 ):
        """

        :param background_image_tuple: (path, label)
        :param instance_results: Array in Form (100, ) (prediction result)
        :param bag_result: Array in Form (1, ) (bag prediction result)
        :param attention_map: Array in Form (100, ) (intermediate bag prediction result)
        :param max_min_prediction: Tuple in Form (max, min) to normalize colorbar
        :return:
        """
        self.background_image = plt.imread(background_image_tuple[0])[300:-300, 300:-300]
        self.image_label = background_image_tuple[1]
        if attention_map is None:
            self.attention_map_resized = np.zeros(self.background_image.shape)
        else:
            self.attention_map_resized = resize(np.minimum(1 - (attention_map / attention_map.max()), 0.8), self.background_image.shape)
        self.instance_prediction_resized = resize(instance_results, self.background_image.shape, preserve_range=True)
        self.bag_result = bag_result
        self.show_only_important_predictions = np.abs(self.instance_prediction_resized) / \
                                               (1.4 * np.max(np.abs(self.instance_prediction_resized)))
        self.max_min_prediction = max_min_prediction

    def create_figure(self, save_at, attention=True):
        fig = plt.figure(figsize=(18, 18))
        plt.axis(False)
        plt.title(f"Ground Truth: {self.image_label}, Predicted: {self.bag_result}")
        plt.imshow(self.background_image, cmap="gray")
        plt.imshow(self.instance_prediction_resized, cmap="seismic", alpha=self.show_only_important_predictions)
        cmin, cmax = self.get_color_limits()
        plt.clim(cmin, cmax)
        plt.colorbar()
        plt.imshow(np.ones(self.background_image.shape), cmap="gray", alpha=self.attention_map_resized)
        #fig.savefig(save_at)
        #plt.close()
        fig.show()

    def get_color_limits(self):
        abs_max = np.max(np.abs(self.max_min_prediction))
        return -abs_max, abs_max

    def bokeh_plot(self):
        pass


if __name__ == "__main__":
    os.chdir("../")
    file_list = ImageDataset.load_file_list("test")
    background_image_path = file_list[0]
    attention_map = np.zeros((10, 10))
    attention_map[4, 6] = 0.1
    attention_map[4, 7] = 0.1
    attention_map[4, 8] = 0.1
    attention_map[3, 6] = 0.3
    attention_map[3, 7] = 0.3
    attention_map[3, 8] = 0.3
    attention_map[2, 6] = 0.1
    attention_map[2, 7] = 0.1
    attention_map[2, 8] = 0.1
    instance_prediction = np.random.randint(-5, 10, (10, 10))
    vis = MilVisualizer(background_image_path, instance_results=instance_prediction, bag_result=2,
                  attention_map=attention_map, max_min_prediction=(-10, 10))
    vis.create_figure()
