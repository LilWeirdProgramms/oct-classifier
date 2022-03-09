import numpy as np
from PIL import Image
from PIL import ImageDraw
import os

class ImageVisualizer:
    """
    Overlays results and an Instance Map over a background image (which is the bag)
    """

    def __init__(self,
                 results = None,
                 info_map: list = None,
                 background_image_path: str = None,
                 image_size=(2044, 2048),
                 instance_size=(28, 23, 89, 73),  #TODO
                 bag_number=0):
        """

        :param results: Large Array of the prediction results of the NN. Probabably size 89*73
        :param info_map:
        :param background_image:
        :param image_size:
        :param instance_size:
        :param bag_number:
        """
        self.info_Map = info_map
        self.results = results
        self.background_image = background_image_path
        self.image_size = image_size
        self.instance_size = instance_size
        self.bag_number = bag_number

    def plot_results_map(self, name=1):
        results_grid = self._create_grid()
        if self.background_image:
            self._preprocess_background()
            results_grid = Image.blend(self.background_image, results_grid, 0.4)
        if any(self.results):
            self._place_results_in_grid(results_grid)
        image_path = f"results/instance_probability_{name}.png"
        results_grid.save(image_path)

    def _place_results_in_grid(self, unplaced_image):
        if not self.info_Map:
            sorted_results = sorted((self.results[:, 0]))
            bound = 10
            upper_bound = sorted_results[-bound]
            lower_bound = sorted_results[10]
            for i, instance_prop in enumerate(self.results[:, 0]):
                instance_position = [i % self.instance_size[2], i // self.instance_size[3]]
                placement = (instance_position[0] * self.instance_size[1] + 2,
                             instance_position[1] * self.instance_size[0])
                color = (0, 255, 0) if instance_prop > upper_bound \
                    else (255, 0, 0) if instance_prop < lower_bound \
                    else (255, 255, 255)
                ImageDraw.Draw(unplaced_image).text(
                    placement,  # Coordinates
                    str(int(instance_prop*10)),
                    #str(int(instance_position[1])),
                    color  # Color
                )
        else:
            for instance_prop, instance_info in zip(self.results, self.info_Map):
                instance_position = instance_info[1]
                placement = (instance_position[1] * self.instance_size[1] + 2,
                             instance_position[0] * self.instance_size[0])
                color = (255, 0, 0) if instance_prop > 0.4 else (255, 255, 255)
                ImageDraw.Draw(unplaced_image).text(
                    placement,  # Coordinates
                    str(instance_prop),
                    color  # Color
                )

    def _create_grid(self):
        """
        Creates Grid with lines belonging to the bottom right instance
        :return: image sized grid with instance sized spacing
        """
        grid = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 80
        grid[:, self.instance_size[1]:-1:self.instance_size[1]] = [255, 0, 0]
        grid[self.instance_size[0]:-1:self.instance_size[0], :] = [255, 0, 0]
        grid = Image.fromarray(grid).convert("RGB")
        return grid

    def _preprocess_background(self):
        self.background_image = Image.open(self.background_image).convert("RGB")

    @staticmethod
    def raw_path_to_image(image_path: str):
        image_ident = ["retina", "png"]
        binary_ident = ["raw", "bin"]
        for a, b in zip(image_ident, binary_ident):
            image_path = image_path.replace(b, a)
        if not os.path.exists(image_path):
            image_path = image_path.replace("retina", "choroid")
            if not os.path.exists(image_path):
                image_path = None
        return image_path
