import BinaryReader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw


class ImageVisualizer:

    def __init__(self, info_map: list):
        self.info_Map = info_map

    def create_image_map(self, info_map, dimensions: BinaryReader.InstanceDim = None):
        # if not dimensions:

        pass

    def plot_results_map(self, results, info_map, overlay_image = None):
        pass

    def create_grid(self, x_spacing, y_spacing, image_size=(2047, 2045)):
        grid = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        grid[:, y_spacing:-1:y_spacing] = [255, 0, 0]  # red horizontal lines
        grid[x_spacing:-1:x_spacing, :] = [255, 0, 0]  # blue vertical lines
        grid = Image.fromarray(grid).convert("RGB")
        return grid

    def preprocess_background(self, background):
        return background.convert("RGB")
