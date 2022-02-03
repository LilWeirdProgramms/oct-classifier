import numpy as np
from PIL import Image
from PIL import ImageDraw


class ImageVisualizer:
    """
    Overlays results and an Instance Map over a background image (which is the bag)
    """

    def __init__(self,
                 results,
                 info_map: list,
                 background_image_path: str = None,
                 image_size=(2047, 2045),
                 instance_size=(23, 28),
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

    def plot_results_map(self):
        results_grid = self._create_grid()
        if self.background_image:
            self._preprocess_background()
            results_grid = Image.blend(self.background_image, results_grid, 0.5)
        self._place_results_in_grid(results_grid)
        results_grid.save("results/instance_probability.png")

    def _place_results_in_grid(self, unplaced_image):
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
        grid = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        grid[:, self.instance_size[1]:-1:self.instance_size[1]] = [255, 0, 0]
        grid[self.instance_size[0]:-1:self.instance_size[0], :] = [255, 0, 0]
        grid = Image.fromarray(grid).convert("RGB")
        return grid

    def _preprocess_background(self):
        self.background_image = Image.open(self.background_image).convert("RGB")
