import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import os


class MilPooling:
    def __init__(self, mil_prediction: np.ndarray, mil_pooling_model_name=None,
                 mil_pooling_model_folder="results/hyperparameter_study/mil/models",
                 mil_pooling_type="weighted"):
        self.instance_prediction = mil_prediction.reshape((-1, 10, 10))
        self.instance_prediction = np.swapaxes(self.instance_prediction, 1, 2).reshape((-1, 100))
        self.model_name = mil_pooling_model_name
        self.model_location = mil_pooling_model_folder
        if mil_pooling_type == "weighted":
            self.conduct_pooling = self.weighted_average
            self.get_attention_weights = self.instance_weights

    # TODO: Make Dataset Output its elements in right order

    def conduct_pooling(self):
        pass

    def weighted_average(self):
        bag_predictions = np.sum(self.instance_prediction, axis=1)
        normalized_bag_predictions = bag_predictions - np.mean(bag_predictions)
        return normalized_bag_predictions

    def instance_weights(self, prediction):
        return np.ones((100, ))

    def shallow_mil_pooling(self):
        model = k.models.load_model(os.path.join(self.model_location, self.model_name))
        bag_prediction = model.predict(self.instance_prediction, batch_size=1, verbose=1).flatten()
        return bag_prediction

    def get_attention_weights(self, data):
        layer_name = 'attention_weights'
        model = k.models.load_model(os.path.join(self.model_location, self.model_name))
        # intermediate_layer_model = k.models.Model(inputs=model.input,
        #                                  outputs=model.get_layer(layer_name).output)
        intermediate_layer_model = k.models.Model(inputs=model.input,
                                            outputs=model.layers[1].output)
        attention_weights = intermediate_layer_model.predict(data)
        #plt.imshow(intermediate_output.reshape((10, 10)), cmap="gray")
        #plt.colorbar()
        return attention_weights

    def save_mil_pooling_model(self, path):
        pass
        # Save at Same Path with mil_pooling_ instead of acc_
