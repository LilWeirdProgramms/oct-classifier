import os
import tensorflow.keras as k
from hyperparameterStudy.image_dataset import ImageDataset
import numpy as np
import Callbacks

class MilPoolingTrainer:

    def __init__(self, mil_dataset, file_list, model_name=None, model_folder="results/hyperparameter_study/mil/models", model=None):
        if model_name:
            full_model_path = os.path.join(model_folder, model_name)
            model = k.models.load_model(full_model_path)
        self.model_name = model_name
        self.model_folder = model_folder
        self.instance_model = model
        self.instance_dataset, self.bag_label = self.create_instance_predictions()
        self.mil_dataset = mil_dataset
        self.file_list = file_list

    def create_instance_predictions(self):
        instance_predictions = self.instance_model.predict(mil_dataset.dataset_train.batch(1), verbose=1)\
            .reshape((-1, 10, 10))
        instance_predictions = np.swapaxes(instance_predictions, 1, 2).reshape((-1, 100))
        for a in instance_predictions:
            print(a.min())
        bag_labels = np.array([label for path, label in file_list])
        return instance_predictions, bag_labels

    def train_model(self):
        model = self.small_attention_mil_pooling_model((100, ))
        model.fit(x=self.instance_dataset, y=self.bag_label,
                  batch_size=16, validation_split=0.2, epochs=1000,
                  callbacks=Callbacks.mil_pooling_callback(self.model_name.replace("acc_", "mil_pooling_")
                                                           .replace("loss_", "mil_pooling_")))

    def shallow_mil_pooling_model(self, input_shape):
        inp = k.layers.Input(shape=input_shape)
        out = k.layers.Dense(1, kernel_regularizer=k.regularizers.L2(), use_bias=False, activation="sigmoid")(inp)
        model = k.Model(inp, out)
        model.summary()
        model.compile(loss=k.losses.BinaryCrossentropy(from_logits=False),
                      optimizer=k.optimizers.Adam(learning_rate=1e-4),
                      metrics=["accuracy"])
        return model

    def small_attention_mil_pooling_model(self, input_shape):
        inp = k.layers.Input(shape=input_shape)
        weights = k.layers.Dense(input_shape[0], kernel_regularizer=k.regularizers.L2(), use_bias=False,
                                 activation="softmax", name='attention_weights')(inp)
        out = k.layers.Dot(axes=1)([inp, weights])
        model = k.Model(inp, out)
        model.summary()
        model.compile(loss=k.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=k.optimizers.Adam(learning_rate=1e-4),
                      metrics=["accuracy"])
        return model

if __name__ == "__main__":
    file_list = ImageDataset.load_file_list("train")
    mil_dataset = ImageDataset(data_list=file_list, validation_split=False, mil=True)
    my_model_names = os.listdir("results/hyperparameter_study/mil/models")
    for model_name in my_model_names:
        #model_name = "acc_ave_pool_selu_lay5_no_drop_little_l2_global_ave_pooling_n32_zeros_augment_noise_second_residual_mil"
        if not "mil_pooling" in model_name:
            mp = MilPoolingTrainer(mil_dataset, file_list, model_name)
            mp.train_model()
