from dataclasses import dataclass

@dataclass
class Hyperparameter:
    downsample: list
    activation: list
    conv_layer: list
    dropout: list
    regularizer: list
    reduction: list
    first_layer: list
    init: list
    augment: list
    noise: list
    repetition: list
    residual: list
    mil: list
    crop: list
    normalize: list
    image_type: list
    label_smoothing: list

