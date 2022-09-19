from hyperparameterStudy.hyper_postprocessing import HyperPostprocessor
from hyperparameterStudy.image_dataset import ImageDataset
from hyperparameterStudy.image_dataset import ImageDataset
from tensorflow.keras.applications import VGG16
import tensorflow as tf
import random
import tensorflow.keras as k
import Callbacks
from PreprocessImageData import PreprocessImageData
from vgg16 import create_vgg_model
from PreprocessData import PreprocessData

# def create_vgg_model(input_shape=(1444, 1448, 3), train_from_layer=10):
#     """
#
#     :param input_shape:
#     :param train_from_layer:
#     :return:
#     """
#     network_input = k.layers.Input(shape=input_shape)
#     model = VGG16(weights='imagenet', input_tensor=network_input, include_top=False)
#     model.summary()
#     feature_extraction_model = k.models.Model(inputs=model.input, outputs=model.layers[train_from_layer].output)
#     feature_extraction_model.trainable = False
#     classify_model = k.models.Model(inputs=model.layers[train_from_layer].output, outputs=model.output)
#
#     pooling_layer = k.layers.GlobalAveragePooling2D()
#     dropout_layer = k.layers.Dropout(0.1)
#     dense_layer1 = k.layers.Dense(64, activation="relu")
#     dense_layer2 = k.layers.Dense(1, activation="linear")
#     model = k.Sequential([feature_extraction_model, classify_model, pooling_layer, dropout_layer, dense_layer1,
#                           dense_layer2])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#     model.summary()
#     return model


def eval_vgg(model_name):

    data_type = "test"
    file_list = PreprocessData.load_file_list(test_or_train=data_type, angio_or_structure="images")
    pid = PreprocessImageData(file_list, rgb=True, crop=0)
    pid.preprocess_data_and_save()
    ds_test = pid.create_dataset_for_calculation()
    ds_test = ds_test.map(lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y))
    hp = HyperPostprocessor(model_name, ds_test, pid.calculation_file_list, results_folder="results/vgg",
                            history_folder="results")
    hp.processing_vgg()


def train_vgg(model_name):
    #file_list = ImageDataset.load_file_list("train")[-140:]
    #dataset = ImageDataset(data_list=file_list, validation_split=True, mil=False, rgb=True)
    data_type = "train"

    file_list = PreprocessData.load_file_list(test_or_train=data_type, angio_or_structure="images")
    pid = PreprocessImageData(file_list, rgb=True, crop=False, data_type=data_type)
    pid.preprocess_data_and_save()
    ds_train, ds_val = pid.create_dataset_for_calculation()
    weights = PreprocessData.calc_weights(ds_train)
    # TODO: class weights


    #ds_train = ds_train.map(lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y))
    #ds_val = ds_val.map(lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y))

    for elem in ds_train.take(1):
        print(elem[1])
        print(elem[0].shape)

    vgg_model = create_vgg_model(input_shape=(2044, 2048, 3))
    vgg_model.fit(ds_train.batch(2),
                  validation_data=ds_val.batch(1),
                  epochs=30,
                  callbacks=Callbacks.vgg_callback(model_name),
                  class_weight=weights)

if __name__ == "__main__":
    name = "final_vgg"
    #train_vgg(name)
    eval_vgg(name)
